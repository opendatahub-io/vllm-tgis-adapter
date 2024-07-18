"""Bits that get monkey patched in."""

from pathlib import Path

import torch
from peft import load_peft_weights
from vllm.prompt_adapter.models import PromptAdapterModel


@classmethod
def from_local_checkpoint(
    cls: "PromptAdapterModel",
    adapter_path: str,
    prompt_adapter_id: int,
    torch_device: str = "cuda",
) -> "PromptAdapterModel":
    """Patch for PromptAdapterModel that supports decoder.pt models."""
    peft_config_path = Path(adapter_path) / "adapter_config.json"
    decoder_pt_path = Path(adapter_path) / "decoder.pt"

    if Path(peft_config_path).exists():
        adapters_weights = load_peft_weights(adapter_path, torch_device)
        prompt_embedding = adapters_weights["prompt_embeddings"].half()
    elif Path(decoder_pt_path).exists():
        # if no PEFT adapter found, load caikit-style adapter from path
        prompt_embedding = torch.load(
            decoder_pt_path, weights_only=True, map_location=torch_device
        ).half()
    else:
        raise ValueError(f"No supported adapter format found at path {adapter_path}")
    num_virtual_tokens = prompt_embedding.shape[0]
    return cls(prompt_adapter_id, num_virtual_tokens, prompt_embedding)


def monkey_patch_prompt_adapter() -> None:
    """Insert our own implementation to support decoder.pt prompts."""
    PromptAdapterModel.from_local_checkpoint = from_local_checkpoint
