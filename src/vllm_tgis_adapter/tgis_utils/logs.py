"""Some methods for producing logs similar to TGIS"""

import logging
from typing import List

from google.protobuf import text_format
from vllm import RequestOutput

from vllm_tgis_adapter.grpc.pb.generation_pb2 import (
    GenerationResponse,
    Parameters,
    StopReason,
)


def log_response(
    inputs: List[str],
    params: Parameters,
    prefix_id: str,
    response: GenerationResponse,
    engine_response: RequestOutput,
    start_time: float,
    kind_log: str,
    method_str: str,
    logger: logging.Logger,
):
    """Logs responses similar to how the TGIS server does"""
    # This time contains both request validation and tokenization
    tokenization_time = engine_response.metrics.arrival_time - start_time
    inference_time = (
        engine_response.metrics.last_token_time
        - engine_response.metrics.first_scheduled_time
    )
    queue_time = engine_response.metrics.time_in_queue
    time_per_token = _safe_div(inference_time, response.generated_token_count)
    total_time = engine_response.metrics.last_token_time - start_time
    output_len = len(response.text)
    short_output = _truncate(response.text, 32)
    short_input = [_truncate(input_, 32) for input_ in inputs]
    input_chars = sum(len(input_) for input_ in inputs)

    paramstr = text_format.MessageToString(params, as_one_line=True)
    span_str = (
        f"{method_str}{{input={short_input} prefix_id={prefix_id} "
        f"input_chars=[{input_chars}] params={paramstr} "
        f"tokenization_time={tokenization_time * 1e3:.2f}ms "
        f"queue_time={queue_time * 1e3:.2f}ms "
        f"inference_time={inference_time * 1e3:.2f}ms "
        f"time_per_token={time_per_token * 1e3:.2f}ms "
        f"total_time={total_time * 1e3:.2f}ms "
        f"input_toks={response.input_token_count}}}"
    )
    stop_reason_str = StopReason.Name(response.stop_reason)

    if response.stop_reason == StopReason.ERROR:
        level = logging.ERROR
    elif response.stop_reason in {StopReason.CANCELLED, StopReason.TOKEN_LIMIT}:
        level = logging.WARNING
    else:
        level = logging.INFO
    logger.log(
        level,
        f"{span_str}: {kind_log} generated "
        f"{response.generated_token_count} tokens before "
        f"{stop_reason_str}, output {output_len} chars: "
        f"{short_output}",
    )


def _truncate(text: str, len_: int) -> bytes:
    """Truncates a string and escapes control characters"""
    text = f"{text:.{len_}}..." if len(text) > len_ else text
    return text.encode("unicode_escape")


def _safe_div(a: float, b: float, *, default: float = 0.0) -> float:
    """Simple safe division with a default answer for divide-by-zero."""
    try:
        return a / b
    except ZeroDivisionError:
        return default
