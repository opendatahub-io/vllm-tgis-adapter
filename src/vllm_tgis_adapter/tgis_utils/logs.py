"""Some methods for producing logs similar to TGIS."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from google.protobuf import text_format

from vllm_tgis_adapter.grpc.pb.generation_pb2 import (
    BatchedGenerationRequest,
    StopReason,
)

if TYPE_CHECKING:
    from vllm.sequence import RequestMetrics

    from vllm_tgis_adapter.grpc.pb.generation_pb2 import (
        GenerationResponse,
        Parameters,
        SingleGenerationRequest,
    )


def log_response(  # noqa: PLR0913
    request: BatchedGenerationRequest | SingleGenerationRequest,
    response: GenerationResponse,
    engine_metrics: RequestMetrics | None,
    start_time: float,
    logger: logging.Logger,
    headers: dict,
    sub_request_num: int = 0,
) -> None:
    if isinstance(request, BatchedGenerationRequest):
        # unary case
        request_count = len(request.requests)
        kind_log = (
            "Request"
            if request_count == 1
            else f"Sub-request {sub_request_num} from batch of {request_count}"
        )
        inputs = [r.text for r in request.requests]
        method_str = "generate"
    else:
        # streaming case
        inputs = [request.request.text]
        kind_log = "Streaming response"
        method_str = "generate_stream"

    _log_response(
        inputs=inputs,
        response=response,
        params=request.params,
        prefix_id=request.prefix_id,
        adapter_id=request.adapter_id,
        engine_metrics=engine_metrics,
        start_time=start_time,
        kind_log=kind_log,
        method_str=method_str,
        logger=logger,
        headers=headers,
    )


def log_error(
    request: BatchedGenerationRequest | SingleGenerationRequest,
    exception_str: str,
    logger: logging.Logger,
) -> None:
    """Log errors similar to how the TGIS server does."""
    # NB: We don't actually log the `Exception` here to match the TGIS behavior
    # of just logging the simple string representation of the error
    param_str = text_format.MessageToString(request.params, as_one_line=True)
    prefix_id = request.prefix_id
    adapter_id = request.adapter_id

    if isinstance(request, BatchedGenerationRequest):
        method_str = "generate"
        inputs = [r.text for r in request.requests]
    else:
        method_str = "generate_stream"
        inputs = [request.request.text]

    short_input = [_truncate(input_, 32) for input_ in inputs]
    input_chars = sum(len(input_) for input_ in inputs)

    span_str = (
        f"{method_str}{{input={short_input} prefix_id={prefix_id} "
        f"adapter_id={adapter_id} input_chars=[{input_chars}] "
        f"params={param_str}"
    )

    logger.error("%s: %s", span_str, exception_str)


def _log_response(  # noqa: PLR0913
    inputs: list[str],
    params: Parameters,
    prefix_id: str,
    adapter_id: str,
    response: GenerationResponse,
    engine_metrics: RequestMetrics | None,
    start_time: float,
    kind_log: str,
    method_str: str,
    logger: logging.Logger,
    headers: dict,
) -> None:
    """Log responses similar to how the TGIS server does."""
    # This time contains both request validation and tokenization
    if engine_metrics is None:
        logger.warning("No engine metrics for request, cannot log timing info")
        tokenization_time = inference_time = queue_time = time_per_token = (
            total_time
        ) = 0.0
    else:
        assert engine_metrics is not None
        assert engine_metrics.first_scheduled_time is not None

        tokenization_time = engine_metrics.arrival_time - start_time
        inference_time = (
            engine_metrics.last_token_time - engine_metrics.first_scheduled_time
        )
        assert engine_metrics.time_in_queue is not None
        queue_time = engine_metrics.time_in_queue

        time_per_token = _safe_div(inference_time, response.generated_token_count)
        total_time = engine_metrics.last_token_time - start_time
    output_len = len(response.text)
    short_output = _truncate(response.text, 32)
    short_input = [_truncate(input_, 32) for input_ in inputs]
    input_chars = sum(len(input_) for input_ in inputs)

    paramstr = text_format.MessageToString(params, as_one_line=True)
    span_str = (
        f"{method_str}{{input={short_input} prefix_id={prefix_id} "
        f"correlation_id={headers.get('x-correlation-id')} "
        f"adapter_id={adapter_id} "
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
        "%s: %s generated %d tokens before %s, output %d chars: %s",
        span_str,
        kind_log,
        response.generated_token_count,
        stop_reason_str,
        output_len,
        short_output,
    )


def _truncate(text: str, len_: int) -> bytes:
    """Truncate a string and escape control characters."""
    text = f"{text:.{len_}}..." if len(text) > len_ else text
    return text.encode("unicode_escape")


def _safe_div(a: float, b: float, *, default: float = 0.0) -> float:
    """Simple safe division with a default answer for divide-by-zero."""  # noqa: D401
    try:
        return a / b
    except ZeroDivisionError:
        return default
