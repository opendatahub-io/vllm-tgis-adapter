from __future__ import annotations

import asyncio
import concurrent.futures
from re import escape as regex_escape

from transformers import PreTrainedTokenizer
from vllm.model_executor.guided_decoding import outlines_decoding
from vllm.model_executor.guided_decoding.outlines_decoding import (
    GuidedDecodingMode,
    _get_logits_processor,
)

# ruff: noqa: TCH002
from vllm.model_executor.guided_decoding.outlines_logits_processors import (
    JSONLogitsProcessor,
    RegexLogitsProcessor,
)

from vllm_tgis_adapter.grpc.pb.generation_pb2 import DecodingParameters


async def get_outlines_guided_decoding_logits_processor(
    decoding_params: DecodingParameters, tokenizer: PreTrainedTokenizer
) -> JSONLogitsProcessor | RegexLogitsProcessor | None:
    """Check for guided decoding parameters.

    Check for guided decoding parameters and get the
    necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    """
    guide, mode = _get_guide_and_mode(decoding_params)
    if not guide:
        return None

    if outlines_decoding.global_thread_pool is None:
        outlines_decoding.global_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2
        )
    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(
        outlines_decoding.global_thread_pool,
        _get_logits_processor,
        guide,
        tokenizer,
        mode,
        None,  # guided_whitespace_pattern - TBD
    )


def _get_guide_and_mode(
    decoding_params: DecodingParameters,
) -> tuple[str, GuidedDecodingMode] | tuple[None, None]:
    guided = decoding_params.WhichOneof("guided")
    if guided is not None:
        if guided == "json_schema":
            return decoding_params.json_schema, GuidedDecodingMode.JSON
        if guided == "regex":
            return decoding_params.regex, GuidedDecodingMode.REGEX
        if guided == "choice":
            choice_list = decoding_params.choice.choices
            if len(choice_list) < 2:
                raise ValueError("Must provide at least two choices")
            # choice just uses regex
            choices = [regex_escape(str(choice)) for choice in choice_list]
            choices_regex = "(" + "|".join(choices) + ")"
            return choices_regex, GuidedDecodingMode.CHOICE
        if guided == "grammar":
            return decoding_params.grammar, GuidedDecodingMode.GRAMMAR
        if decoding_params.format == DecodingParameters.JSON:
            return outlines_decoding.JSON_GRAMMAR, GuidedDecodingMode.GRAMMAR
    return None, None
