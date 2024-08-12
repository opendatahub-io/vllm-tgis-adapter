from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vllm.utils import FlexibleArgumentParser

from vllm_tgis_adapter.logging import init_logger
from vllm_tgis_adapter.tgis_utils import hub

logger = init_logger(__name__)

if TYPE_CHECKING:
    import argparse


def tgis_cli(args: argparse.Namespace) -> None:
    if args.command == "download-weights":
        download_weights(
            args.model_name,
            args.revision,
            args.token,
            args.extension,
            args.auto_convert,
        )
    elif args.command == "convert-to-safetensors":
        convert_to_safetensors(args.model_name, args.revision)
    elif args.command == "convert-to-fast-tokenizer":
        convert_to_fast_tokenizer(args.model_name, args.revision, args.output_path)


def download_weights(
    model_name: str,
    revision: str | None = None,
    token: str | None = None,
    extension: str = ".safetensors",
    auto_convert: bool | None = None,
) -> None:
    if auto_convert is None:
        auto_convert = True

    logger.info(extension)
    meta_exts = [".json", ".py", ".model", ".md"]

    extensions = extension.split(",")

    if len(extensions) == 1 and extensions[0] not in meta_exts:
        extensions.extend(meta_exts)

    files = hub.download_weights(
        model_name, extensions, revision=revision, auth_token=token
    )

    if auto_convert and ".safetensors" in extensions:
        if not hub.local_weight_files(
            hub.get_model_path(model_name, revision), ".safetensors"
        ):
            if ".bin" not in extensions:
                logger.info(
                    ".safetensors weights not found, \
                    downloading pytorch weights to convert..."
                )
                hub.download_weights(
                    model_name, ".bin", revision=revision, auth_token=token
                )

            logger.info(
                ".safetensors weights not found, \
                    converting from pytorch weights..."
            )
            convert_to_safetensors(model_name, revision)
        elif not any(f.endswith(".safetensors") for f in files):
            logger.info(
                ".safetensors weights not found on hub, \
                    but were found locally. Remove them first to re-convert"
            )
    if auto_convert:
        convert_to_fast_tokenizer(model_name, revision)


def convert_to_safetensors(
    model_name: str,
    revision: str | None = None,
) -> None:
    # Get local pytorch file paths
    model_path = hub.get_model_path(model_name, revision)
    local_pt_files = hub.local_weight_files(model_path, ".bin")
    local_pt_index_files = hub.local_index_files(model_path, ".bin")
    if len(local_pt_index_files) > 1:
        logger.info(
            "Found more than one .bin.index.json file: %s", local_pt_index_files
        )
        return
    if not local_pt_files:
        logger.info("No pytorch .bin files found to convert")
        return

    local_pt_files = [Path(f) for f in local_pt_files]
    local_pt_index_file = local_pt_index_files[0] if local_pt_index_files else None

    # Safetensors final filenames
    local_st_files = [
        p.parent / f"{p.stem.removeprefix('pytorch_')}.safetensors"
        for p in local_pt_files
    ]

    if any(Path.exists(p) for p in local_st_files):
        logger.info(
            "Existing .safetensors weights found, remove them first to reconvert"
        )
        return

    try:
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            model_name,
            revision=revision,
        )
        architecture = config.architectures[0]

        class_ = getattr(transformers, architecture)

        # Name for this variable depends on transformers version
        discard_names = getattr(class_, "_tied_weights_keys", [])
        discard_names.extend(getattr(class_, "_keys_to_ignore_on_load_missing", []))

    except TypeError:
        discard_names = []

    if local_pt_index_file:
        local_pt_index_file = Path(local_pt_index_file)
        st_prefix = local_pt_index_file.stem.removeprefix("pytorch_").removesuffix(
            ".bin.index"
        )
        local_st_index_file = (
            local_pt_index_file.parent / f"{st_prefix}.safetensors.index.json"
        )

        if Path.exists(local_st_index_file):
            logger.info(
                "Existing .safetensors.index.json file found, remove it first to \
                    reconvert"
            )
            return

        hub.convert_index_file(
            local_pt_index_file, local_st_index_file, local_pt_files, local_st_files
        )

    # Convert pytorch weights to safetensors
    hub.convert_files(local_pt_files, local_st_files, discard_names)


def convert_to_fast_tokenizer(
    model_name: str,
    revision: str | None = None,
    output_path: str | None = None,
) -> None:
    # Check for existing "tokenizer.json"
    model_path = hub.get_model_path(model_name, revision)

    if Path.exists(Path(model_path) / "tokenizer.json"):
        logger.info("Model %s already has a fast tokenizer", model_name)
        return

    if output_path is not None and not Path.isdir(output_path):
        logger.info("Output path %s must exist and be a directory", output_path)
        return
    output_path = model_path

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, revision=revision
    )
    tokenizer.save_pretrained(output_path)

    logger.info("Saved tokenizer to %s", output_path)


def cli() -> None:
    parser = FlexibleArgumentParser(description="vLLM CLI")
    subparsers = parser.add_subparsers(required=True)

    download_weights_parser = subparsers.add_parser(
        "download-weights",
        help=("Download the weights of a given model"),
        usage="model-util download-weights <model_name> [options]",
    )
    download_weights_parser.add_argument("model_name")
    download_weights_parser.add_argument("--revision")
    download_weights_parser.add_argument("--token")
    download_weights_parser.add_argument("--extension", default=".safetensors")
    download_weights_parser.add_argument("--auto_convert", default=True)
    download_weights_parser.set_defaults(
        dispatch_function=tgis_cli, command="download-weights"
    )

    convert_to_safetensors_parser = subparsers.add_parser(
        "convert-to-safetensors",
        help=("Convert model weights to safetensors"),
        usage="model-util convert-to-safetensors <model_name> [options]",
    )
    convert_to_safetensors_parser.add_argument("model_name")
    convert_to_safetensors_parser.add_argument("--revision")
    convert_to_safetensors_parser.set_defaults(
        dispatch_function=tgis_cli, command="convert-to-safetensors"
    )

    convert_to_fast_tokenizer_parser = subparsers.add_parser(
        "convert-to-fast-tokenizer",
        help=("Convert to fast tokenizer"),
        usage="model-util convert-to-fast-tokenizer <model_name> [options]",
    )
    convert_to_fast_tokenizer_parser.add_argument("model_name")
    convert_to_fast_tokenizer_parser.add_argument("--revision")
    convert_to_fast_tokenizer_parser.add_argument("--output_path")
    convert_to_fast_tokenizer_parser.set_defaults(
        dispatch_function=tgis_cli, command="convert-to-fast-tokenizer"
    )

    args = parser.parse_args()
    # One of the sub commands should be executed.
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
