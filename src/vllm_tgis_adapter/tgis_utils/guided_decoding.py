from __future__ import annotations

import asyncio
import concurrent.futures
from re import escape as regex_escape
from typing import TYPE_CHECKING

from vllm import __version_tuple__ as vllm_version

if vllm_version <= (0, 10, 0):
    from vllm.model_executor.guided_decoding import outlines_decoding
    from vllm.model_executor.guided_decoding.outlines_decoding import (
        GuidedDecodingMode,
        _get_logits_processor,
    )
else:
    from vllm.sampling_params import GuidedDecodingParams

from vllm_tgis_adapter.grpc.pb.generation_pb2 import DecodingParameters

if TYPE_CHECKING:
    from vllm.model_executor.guided_decoding.outlines_logits_processors import (
        JSONLogitsProcessor,
        RegexLogitsProcessor,
    )
    from vllm.transformers_utils.tokenizer import AnyTokenizer


async def get_outlines_guided_decoding_logits_processor(
    decoding_params: DecodingParameters, tokenizer: AnyTokenizer
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


# vllm<=0.10.0 (mostly broken)
def _get_guide_and_mode(
    decoding_params: DecodingParameters,
) -> tuple[str, GuidedDecodingMode] | tuple[None, None]:
    if not (guided := decoding_params.WhichOneof("guided")):
        return None, None

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

    raise ValueError(f"{guided=}")


# vllm>=0.10.1
def get_guided_decoding_params(
    decoding_params: DecodingParameters,
) -> GuidedDecodingParams | None:
    if not (guided := decoding_params.WhichOneof("guided")):
        return None

    if guided == "json_schema":
        return GuidedDecodingParams(json=decoding_params.json_schema)

    if guided == "regex":
        return GuidedDecodingParams(regex=decoding_params.regex)

    if guided == "choice":
        choice_list = decoding_params.choice.choices
        if len(choice_list) < 2:
            raise ValueError("Must provide at least two choices")
        return GuidedDecodingParams(choice=list(choice_list))

    if guided == "grammar":
        return GuidedDecodingParams(grammar=decoding_params.grammar)

    if decoding_params.format == DecodingParameters.JSON:
        return GuidedDecodingParams(json_object=True)

    raise ValueError(f"{guided=}")
