"""Some methods for producing logs similar to TGIS."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import suppress
from typing import TYPE_CHECKING

import cachetools

from vllm_tgis_adapter.logging import init_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from vllm import PromptType, RequestOutput, SamplingParams
    from vllm.engine.protocol import EngineClient
    from vllm.sequence import RequestMetrics

logger = init_logger(__name__)

# Blackboard for storing correlation ids when they come into the servers, so
# that they can be added to log messages here.
# TTLCache with a max size is used to make sure everything is cleaned up and we
# don't leak memory.
_REQUEST_ID_TO_CORRELATION_ID = cachetools.TTLCache(maxsize=2048, ttl=600)


def set_correlation_id(request_id: str, correlation_id: str) -> None:
    _REQUEST_ID_TO_CORRELATION_ID[request_id] = correlation_id


def get_correlation_id(request_id: str) -> str | None:
    correlation_id = _REQUEST_ID_TO_CORRELATION_ID.get(request_id)
    if not correlation_id:
        # http server will format the request id like:
        # {method_str}-{base_request_id}-{batch_index}
        # So we can try stripping off leading and trailing `-` clauses
        request_id = "-".join(request_id.split("-")[1:-1])
        correlation_id = _REQUEST_ID_TO_CORRELATION_ID.get(request_id)
    return correlation_id


def add_logging_wrappers(engine: EngineClient) -> None:
    """Inject request, response, and error logs into the engine.

    We log this way so that all entrypoints are logged consistently, whether the
    original request is from the grpc server or the http server, and regardless
    of which endpoint is actually used.
    """
    old_generate_fn = engine.generate

    @functools.wraps(old_generate_fn)
    async def generate_with_logging(*args, **kwargs) -> AsyncGenerator[RequestOutput]:  # noqa: ANN003 ANN002
        start_time = time.time()

        # NB: coupled to EngineClient.generate() api
        prompt = _get_arg("prompt", 0, *args, **kwargs)
        sampling_params = _get_arg("sampling_params", 1, *args, **kwargs)
        request_id = _get_arg("request_id", 2, *args, **kwargs)
        lora_request = _get_arg("lora_request", 3, *args, **kwargs)
        prompt_adapter_request = _get_arg("prompt_adapter_request", 5, *args, **kwargs)

        correlation_id = get_correlation_id(request_id=request_id)
        adapter_id = None
        if lora_request:
            adapter_id = lora_request.adapter_id
        elif prompt_adapter_request:
            adapter_id = prompt_adapter_request.prompt_adapter_id

        # Log the request
        with suppress(BaseException):
            _log_request(
                prompt=prompt,
                params=sampling_params,
                request_id=request_id,
                correlation_id=correlation_id,
                adapter_id=adapter_id,
            )

        # Run the generate method
        last = None
        try:
            async for response in old_generate_fn(*args, **kwargs):
                last = response
                yield response
        except asyncio.CancelledError:
            _log_cancellation(
                request_id=request_id,
                correlation_id=correlation_id,
            )
            raise
        except BaseException as e:
            # Log any error
            _log_error(
                request_id=request_id,
                correlation_id=correlation_id,
                exception_str=str(e),
            )
            raise

        if last:
            # Log the response
            with suppress(BaseException):
                _log_response(
                    request_id=request_id,
                    correlation_id=correlation_id,
                    response=last,
                    engine_metrics=last.metrics,
                    start_time=start_time,
                )

    engine.generate = generate_with_logging


def _log_error(request_id: str, correlation_id: str, exception_str: str) -> None:
    logger.error(
        "Request failed: request_id=%s correlation_id=%s error=%s",
        request_id,
        correlation_id,
        exception_str,
    )


def _log_cancellation(request_id: str, correlation_id: str) -> None:
    logger.info(
        "Request cancelled: request_id=%s correlation_id=%s",
        request_id,
        correlation_id,
    )


def _log_request(
    request_id: str,
    params: SamplingParams,
    adapter_id: str,
    correlation_id: str,
    prompt: PromptType,
) -> None:
    if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
        input_tokens = f" input_tokens={len(prompt['prompt_token_ids'])},"
    else:
        input_tokens = ""
    logger.info(
        "Processing request: {request_id=%s, correlation_id=%s, adapter_id=%s, "
        "%sparams=%s}",
        request_id,
        correlation_id,
        adapter_id,
        input_tokens,
        params,
    )


def _log_response(
    request_id: str,
    correlation_id: str,
    response: RequestOutput,
    engine_metrics: RequestMetrics | None,
    start_time: float,
) -> None:
    """Log responses similar to how the TGIS server does."""
    if len(response.outputs) == 0:
        # Nothing to log about
        return

    generated_tokens = len(response.outputs[0].token_ids)
    if engine_metrics is None:
        logger.warning("No engine metrics for request, cannot log timing info")
        inference_time = queue_time = time_per_token = total_time = 0.0
    else:
        assert engine_metrics is not None
        assert engine_metrics.first_scheduled_time is not None

        inference_time = (
            engine_metrics.last_token_time - engine_metrics.first_scheduled_time
        )
        assert engine_metrics.time_in_queue is not None
        queue_time = engine_metrics.time_in_queue

        time_per_token = _safe_div(inference_time, generated_tokens)
        total_time = engine_metrics.last_token_time - start_time
    output_len = len(response.outputs[0].text)

    stop_reason_str = response.outputs[0].finish_reason

    if stop_reason_str == "abort":
        level = logging.WARNING
    else:
        level = logging.INFO
    logger.log(
        level,
        "Finished processing request: {request_id=%s, correlation_id=%s}. "
        "Timing info: {queue_time=%.2fms, inference_time=%.2fms, "
        "time_per_token=%.2fms, total_time=%.2fms}. "
        "Generated %d tokens before finish reason: %s, output %d chars",
        request_id,
        correlation_id,
        queue_time * 1e3,
        inference_time * 1e3,
        time_per_token * 1e3,
        total_time * 1e3,
        generated_tokens,
        stop_reason_str,
        output_len,
    )


def _safe_div(a: float, b: float, *, default: float = 0.0) -> float:
    """Simple safe division with a default answer for divide-by-zero."""  # noqa: D401
    try:
        return a / b
    except ZeroDivisionError:
        return default


def _get_arg(name: str, pos: int, *args, **kwargs) -> object:  # noqa: ANN003 ANN002
    """Get an argument from either position or keyword arguments."""
    if len(args) > pos:
        return args[pos]
    if name in kwargs:
        return kwargs[name]
    return None
