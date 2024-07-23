import concurrent
import datetime
import glob
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import torch
from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
from huggingface_hub.utils import LocalEntryNotFoundError
from safetensors.torch import (_find_shared_tensors, _is_complete, load_file,
                               save_file)
from tqdm import tqdm

TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE") == "true"
logger = logging.getLogger(__name__)


def weight_hub_files(model_name,
                     extension=".safetensors",
                     revision=None,
                     auth_token=None):
    """Get the safetensors filenames on the hub"""
    exts = [extension] if isinstance(extension, str) else extension
    api = HfApi()
    info = api.model_info(model_name, revision=revision, token=auth_token)
    filenames = [
        s.rfilename for s in info.siblings if any(
            s.rfilename.endswith(ext) and len(s.rfilename.split("/")) == 1
            and "arguments" not in s.rfilename and "args" not in s.rfilename
            and "training" not in s.rfilename for ext in exts)
    ]
    return filenames


def weight_files(model_name, extension=".safetensors", revision=None):
    """Get the local safetensors filenames"""
    filenames = weight_hub_files(model_name, extension)
    files = []
    for filename in filenames:
        cache_file = try_to_load_from_cache(model_name,
                                            filename=filename,
                                            revision=revision)
        if cache_file is None:
            raise LocalEntryNotFoundError(
                f"File {filename} of model {model_name} not found in "
                f"{os.getenv('HUGGINGFACE_HUB_CACHE', 'the local cache')}. "
                f"Please run `vllm \
                    download-weights {model_name}` first.")
        files.append(cache_file)

    return files


def download_weights(model_name,
                     extension=".safetensors",
                     revision=None,
                     auth_token=None):
    """Download the safetensors files from the hub"""
    filenames = weight_hub_files(model_name,
                                 extension,
                                 revision=revision,
                                 auth_token=auth_token)

    download_function = partial(
        hf_hub_download,
        repo_id=model_name,
        local_files_only=False,
        revision=revision,
        token=auth_token,
    )

    print(f"Downloading {len(filenames)} files for model {model_name}")
    executor = ThreadPoolExecutor(max_workers=5)
    futures = [
        executor.submit(download_function, filename=filename)
        for filename in filenames
    ]
    files = [
        future.result()
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures))
    ]

    return files


def get_model_path(model_name: str, revision: Optional[str] = None):
    """Get path to model dir in local huggingface hub (model) cache"""
    config_file = "config.json"
    err = None
    try:
        config_path = try_to_load_from_cache(
            model_name,
            config_file,
            cache_dir=os.getenv("TRANSFORMERS_CACHE"
                                ),  # will fall back to HUGGINGFACE_HUB_CACHE
            revision=revision,
        )
        if config_path is not None:
            return config_path.removesuffix(f"/{config_file}")
    except ValueError as e:
        err = e

    if os.path.isfile(f"{model_name}/{config_file}"):
        return model_name  # Just treat the model name as an explicit model path

    if err is not None:
        raise err

    raise ValueError(
        f"Weights not found in local cache for model {model_name}")


def local_weight_files(model_path: str, extension=".safetensors"):
    """Get the local safetensors filenames"""
    ext = "" if extension is None else extension
    return glob.glob(f"{model_path}/*{ext}")


def local_index_files(model_path: str, extension=".safetensors"):
    """Get the local .index.json filename"""
    ext = "" if extension is None else extension
    return glob.glob(f"{model_path}/*{ext}.index.json")


def _remove_duplicate_names(
    state_dict: Dict[str, torch.Tensor],
    *,
    preferred_names: List[str] = None,
    discard_names: List[str] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        # _find_shared_tensors returns a list of sets of names of tensors that
        # have the same data, including sets with one element that aren't shared
        if len(shared) == 1:
            continue

        complete_names = set(
            [name for name in shared if _is_complete(state_dict[name])])
        if not complete_names:
            raise RuntimeError(f"Error while trying to find names to remove \
                    to save state dict, but found no suitable name to \
                        keep for saving amongst: {shared}. None is covering \
                            the entire storage.Refusing to save/load the model \
                                since you could be storing much more \
                                    memory than needed. Please refer to\
            https://huggingface.co/docs/safetensors/torch_shared_tensors \
                                            for more information. \
                                                Or open an issue.")

        keep_name = sorted(list(complete_names))[0]

        # Mechanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


def convert_file(pt_file: Path, sf_file: Path, discard_names: List[str]):
    """
    Convert a pytorch file to a safetensors file
    This will remove duplicate tensors from the file.

    Unfortunately, this might not respect *transformers* convention.
    Forcing us to check for potentially different keys during load when looking
    for specific tensors (making tensor sharing explicit).
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

    dirname = os.path.dirname(sf_file)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_file, metadata=metadata)
    reloaded = load_file(sf_file)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def convert_index_file(source_file: Path, dest_file: Path,
                       pt_files: List[Path], sf_files: List[Path]):
    weight_file_map = {s.name: d.name for s, d in zip(pt_files, sf_files)}

    logger.info(
        "Converting pytorch .bin.index.json files to .safetensors.index.json")
    with open(source_file, "r") as f:
        index = json.load(f)

    index["weight_map"] = {
        k: weight_file_map[v]
        for k, v in index["weight_map"].items()
    }

    with open(dest_file, "w") as f:
        json.dump(index, f, indent=4)


def convert_files(pt_files: List[Path],
                  sf_files: List[Path],
                  discard_names: List[str] = None):
    assert len(pt_files) == len(sf_files)

    # Filter non-inference files
    pairs = [
        p for p in zip(pt_files, sf_files) if not any(s in p[0].name for s in [
            "arguments",
            "args",
            "training",
            "optimizer",
            "scheduler",
            "index",
        ])
    ]

    N = len(pairs)

    if N == 0:
        logger.warning("No pytorch .bin weight files found to convert")
        return

    logger.info("Converting %d pytorch .bin files to .safetensors...", N)

    for i, (pt_file, sf_file) in enumerate(pairs):
        file_count = (i + 1) / N
        logger.info('Converting: [%d] "$s"', file_count, pt_file.name)
        start = datetime.datetime.now()
        convert_file(pt_file, sf_file, discard_names)
        elapsed = datetime.datetime.now() - start
        logger.info('Converted: [%d] "%s" -- Took: %d', file_count,
                    sf_file.name, elapsed)
