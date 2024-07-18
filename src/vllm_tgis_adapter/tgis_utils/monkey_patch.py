"""Bits that get monkey patched in."""

from pathlib import Path

import torch
from peft.utils import load_peft_weights
from vllm.config import PromptAdapterConfig
from vllm.prompt_adapter.models import PromptAdapterModel


@classmethod
def from_local_checkpoint(  # noqa: PLR0913
    cls: "PromptAdapterModel",
    adapter_path: str,
    prompt_adapter_id: int,
    num_virtual_tokens: int,
    config: PromptAdapterConfig,
    device: str = "cuda",
) -> "PromptAdapterModel":
    """Patch for PromptAdapterModel that supports decoder.pt models."""
    if num_virtual_tokens > config.max_prompt_adapter_token:
        raise ValueError(
            f"num_virtual_tokens ({num_virtual_tokens}) should be <= "
            f"max_prompt_adapter_token({config.max_prompt_adapter_token})"
        )

    peft_config_path = Path(adapter_path) / "adapter_config.json"
    decoder_pt_path = Path(adapter_path) / "decoder.pt"

    if Path(peft_config_path).exists():
        adapters_weights = load_peft_weights(adapter_path, device)
        prompt_embedding = adapters_weights["prompt_embeddings"].to(
            config.prompt_adapter_dtype
        )
    elif Path(decoder_pt_path).exists():
        # if no PEFT adapter found, load caikit-style adapter from path
        prompt_embedding = torch.load(
            decoder_pt_path, weights_only=True, map_location=device
        ).half()
    else:
        raise ValueError(f"No supported adapter format found at path {adapter_path}")
    num_virtual_tokens = prompt_embedding.shape[0]
    return cls(prompt_adapter_id, num_virtual_tokens, prompt_embedding)


def monkey_patch_prompt_adapter() -> None:
    """Insert our own implementation to support decoder.pt prompts."""
    PromptAdapterModel.from_local_checkpoint = from_local_checkpoint
