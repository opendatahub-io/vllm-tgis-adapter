from __future__ import annotations

import concurrent
import datetime
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
from huggingface_hub.utils import LocalEntryNotFoundError
from safetensors.torch import _remove_duplicate_names, load_file, save_file
from tqdm import tqdm

logger = logging.getLogger(__name__)


def weight_hub_files(
    model_name: str,
    extension: str = ".safetensors",
    revision: str | None = None,
    auth_token: str | None = None,
) -> list:
    """Get the safetensors filenames on the hub."""
    exts = [extension] if isinstance(extension, str) else extension
    api = HfApi()
    info = api.model_info(model_name, revision=revision, token=auth_token)
    filenames = [
        s.rfilename
        for s in info.siblings
        if any(
            s.rfilename.endswith(ext)
            and len(s.rfilename.split("/")) == 1
            and "arguments" not in s.rfilename
            and "args" not in s.rfilename
            and "training" not in s.rfilename
            for ext in exts
        )
    ]
    return filenames


def weight_files(
    model_name: str, extension: str = ".safetensors", revision: str | None = None
) -> list:
    """Get the local safetensors filenames."""
    filenames = weight_hub_files(model_name, extension)
    files = []
    for filename in filenames:
        cache_file = try_to_load_from_cache(
            model_name, filename=filename, revision=revision
        )
        if cache_file is None:
            raise LocalEntryNotFoundError(
                f"File {filename} of model {model_name} not found in "
                f"{os.getenv('HUGGINGFACE_HUB_CACHE', 'the local cache')}. "
                f"Please run `vllm \
                    download-weights {model_name}` first."
            )
        files.append(cache_file)

    return files


def download_weights(
    model_name: str,
    extension: str = ".safetensors",
    revision: str | None = None,
    auth_token: str | None = None,
) -> list:
    """Download the safetensors files from the hub."""
    filenames = weight_hub_files(
        model_name, extension, revision=revision, auth_token=auth_token
    )

    download_function = partial(
        hf_hub_download,
        repo_id=model_name,
        local_files_only=False,
        revision=revision,
        token=auth_token,
    )

    logger.info("Downloading {len(filenames)} files for model {model_name}")
    executor = ThreadPoolExecutor(max_workers=5)
    futures = [
        executor.submit(download_function, filename=filename) for filename in filenames
    ]
    files = [
        future.result()
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures))
    ]

    return files


def get_model_path(model_name: str, revision: str | None = None) -> str:
    """Get path to model dir in local huggingface hub (model) cache."""
    config_file = "config.json"
    err = None
    try:
        config_path = try_to_load_from_cache(
            model_name,
            config_file,
            cache_dir=os.getenv(
                "TRANSFORMERS_CACHE"
            ),  # will fall back to HUGGINGFACE_HUB_CACHE
            revision=revision,
        )
        if config_path is not None:
            return config_path.removesuffix(f"/{config_file}")
    except ValueError as e:
        err = e

    if Path.isfile(f"{model_name}/{config_file}"):
        return model_name  # Just treat the model name as an explicit model path

    if err is not None:
        raise err

    raise ValueError(f"Weights not found in local cache for model {model_name}")


def local_weight_files(model_path: str, extension: str = ".safetensors") -> list:
    """Get the local safetensors filenames."""
    ext = "" if extension is None else extension
    return Path.glob(f"{model_path}/*{ext}")


def local_index_files(model_path: str, extension: str = ".safetensors") -> list:
    """Get the local .index.json filename."""
    ext = "" if extension is None else extension
    return Path.glob(f"{model_path}/*{ext}.index.json")


def convert_file(pt_file: Path, sf_file: Path, discard_names: list[str]) -> None:
    """Convert a pytorch file to a safetensors file.

    This will remove duplicate tensors from the file. Unfortunately, this might not
    respect *transformers* convention forcing us to check for potentially different
    keys during load when looking for specific tensors (making tensor sharing explicit).
    """
    loaded = torch.load(pt_file, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    to_removes = _remove_duplicate_names(loaded, discard_names=discard_names)

    metadata = {"format": "pt"}
    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del loaded[to_remove]
    # Force tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = Path.parent(sf_file)
    Path(dirname).mkdir(parents=True)
    save_file(loaded, sf_file, metadata=metadata)
    reloaded = load_file(sf_file)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def convert_index_file(
    source_file: Path, dest_file: Path, pt_files: list[Path], sf_files: list[Path]
) -> None:
    weight_file_map = {s.name: d.name for s, d in zip(pt_files, sf_files)}

    logger.info("Converting pytorch .bin.index.json files to .safetensors.index.json")
    with open(source_file) as f:
        index = json.load(f)

    index["weight_map"] = {
        k: weight_file_map[v] for k, v in index["weight_map"].items()
    }

    with open(dest_file, "w") as f:
        json.dump(index, f, indent=4)


def convert_files(
    pt_files: list[Path], sf_files: list[Path], discard_names: list[str] | None = None
) -> None:
    assert len(pt_files) == len(sf_files)

    # Filter non-inference files
    pairs = [
        p
        for p in zip(pt_files, sf_files)
        if not any(
            s in p[0].name
            for s in [
                "arguments",
                "args",
                "training",
                "optimizer",
                "scheduler",
                "index",
            ]
        )
    ]

    n = len(pairs)

    if n == 0:
        logger.warning("No pytorch .bin weight files found to convert")
        return

    logger.info("Converting %d pytorch .bin files to .safetensors...", n)

    for i, (pt_file, sf_file) in enumerate(pairs):
        file_count = (i + 1) / n
        logger.info('Converting: [%d] "%s"', file_count, pt_file.name)
        start = datetime.datetime.now(tz=datetime.UTC)
        convert_file(pt_file, sf_file, discard_names)
        elapsed = datetime.datetime.now(tz=datetime.UTC) - start
        logger.info(
            'Converted: [%d] "%s" -- Took: %d', file_count, sf_file.name, elapsed
        )
