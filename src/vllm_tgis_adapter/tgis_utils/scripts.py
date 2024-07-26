# The CLI entrypoint to vLLM.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import Path
from vllm.model_executor.model_loader.weight_utils import convert_bin_to_safetensor_file
from vllm.scripts import registrer_signal_handlers
from vllm.utils import FlexibleArgumentParser

from vllm_tgis_adapter.tgis_utils import hub

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import argparse


def tgis_cli(args: argparse.Namespace) -> None:
    registrer_signal_handlers()

    if args.command == "download-weights":
        download_weights(
            args.model_name,
            args.revision,
            args.token,
            args.extension,
            args.auto_convert,
        )
    elif args.command == "convert-to-safetensors":
        convert_bin_to_safetensor_file(args.model_name, args.revision)
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
            convert_bin_to_safetensor_file(model_name, revision)
        elif not any(f.endswith(".safetensors") for f in files):
            logger.info(
                ".safetensors weights not found on hub, \
                    but were found locally. Remove them first to re-convert"
            )
    if auto_convert:
        convert_to_fast_tokenizer(model_name, revision)


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

    if output_path is not None:
        if not Path.isdir(output_path):
            logger.info("Output path %s must exist and be a directory", output_path)
            return
    else:
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
        usage="vllm download-weights <model_name> [options]",
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
        usage="vllm convert-to-safetensors <model_name> [options]",
    )
    convert_to_safetensors_parser.add_argument("model_name")
    convert_to_safetensors_parser.add_argument("--revision")
    convert_to_safetensors_parser.set_defaults(
        dispatch_function=tgis_cli, command="convert-to-safetensors"
    )

    convert_to_fast_tokenizer_parser = subparsers.add_parser(
        "convert-to-fast-tokenizer",
        help=("Convert to fast tokenizer"),
        usage="vllm convert-to-fast-tokenizer <model_name> [options]",
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
