from __future__ import annotations

import torch
from transformers.generation.logits_process import TypicalLogitsWarper


class TypicalLogitsWarperWrapper:
    def __init__(self, mass: float):
        self.warper = TypicalLogitsWarper(mass=mass)

    def __call__(
        self,
        token_ids: list[int],  # noqa: ARG002
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # transformers warpers assume tensors of shape (batch_size, vocab_size)
        # and the typical warper doesn't use input_ids
        return self.warper(
            input_ids=None,
            scores=logits.reshape(1, -1),
        ).flatten()


class ExpDecayLengthPenaltyWarper:
    def __init__(
        self,
        length_penalty: tuple[int, float],
        eos_token_id: int,
    ):
        self.start, self.penalty = length_penalty
        self.eos_token_id = eos_token_id

    def __call__(
        self,
        token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        tokens_past = max(0, len(token_ids) - self.start)
        p_factor = pow(self.penalty, tokens_past)
        if p_factor != 1:
            eos_logit = logits[self.eos_token_id]
            # To support negative logits we compute the penalty of the
            # absolute value and add to the original logit
            logits[self.eos_token_id] = eos_logit + torch.abs(eos_logit) * (
                p_factor - 1
            )
        return logits
