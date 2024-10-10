# This file exists as a separate CLI just because we want to give
# users the ability to be able to run it independently without
# having to install vllm as a dependency
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def cli() -> None:
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        "-i",
        help="directory to find the decoder.pt file in",
        required=True,
    )
    parser.add_argument(
        "--output-dir", "-o", help="directory to write the adapter model to"
    )

    args = parser.parse_args()
    convert_pt_to_peft(input_dir=args.input_dir, output_dir=args.output_dir)


def convert_pt_to_peft(input_dir: str, output_dir: str) -> None:
    # read decoder.pt file
    decoder_pt_path = Path(input_dir) / "decoder.pt"
    if not decoder_pt_path.exists():
        raise ValueError(f"No decoder.pt model found in path {decoder_pt_path}")

    # error if encoder.pt file exists
    encoder_pt_path = Path(input_dir) / "encoder.pt"
    if encoder_pt_path.exists():
        raise ValueError(
            f"encoder.pt model found in path {encoder_pt_path}, \
            encoder-decoder models are not yet supported, sorry!"
        )

    # check output dir
    if output_dir is None:
        output_path = Path(input_dir)

    else:
        output_path = Path(output_dir)

    if not output_path.exists():
        print(  # noqa: T201
            f"output path {output_path} doesn't exist, \
              creating it now..."
        )
        output_path.mkdir(parents=True, exist_ok=True)

    # error if output_dir is file
    if output_path.is_file():
        raise ValueError(f"File found instead of dir {output_path}")

    # load tensors from decoder.pt and save to .safetensors
    decoder_tensors = torch.load(decoder_pt_path, weights_only=True)
    decoder_tensors.requires_grad = False  ## TODO: figure out if needed?
    save_file(
        {"prompt_embeddings": decoder_tensors},
        output_path / "adapter_model.safetensors",
    )

    # write adapter_config.json file
    adapter_config = {
        "num_virtual_tokens": decoder_tensors.shape[0],
        "peft_type": "PROMPT_TUNING",
        "base_model_name_or_path": "this-is-a/temporary-conversion",
    }

    with open(output_path / "adapter_config.json", "w") as config_file:
        import json

        json.dump(adapter_config, config_file, indent=4)


if __name__ == "__main__":
    cli()
