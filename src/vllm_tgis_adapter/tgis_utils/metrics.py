"""Implements the logging for all tgi_* metrics for compatibility with TGIS opsviz."""

import time
from enum import StrEnum, auto

from prometheus_client import Counter, Gauge, Histogram
from vllm import RequestOutput
from vllm.engine.metrics import Stats

try:
    from vllm.engine.metrics import StatLoggerBase
except ImportError:
    # vllm<=0.5.1
    from vllm.engine.metrics import StatLogger as StatLoggerBase

from vllm_tgis_adapter.grpc.pb.generation_pb2 import (
    BatchedTokenizeRequest,
    BatchedTokenizeResponse,
)

_duration_buckets = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    20,
    50,
]


class FailureReasonLabel(StrEnum):
    VALIDATION = auto()  # request validation failed
    CANCELLED = auto()  # TODO: cancellation handling not implemented
    CONC_LIMIT = auto()  # TODO: is this applicable for vLLM?
    OOM = auto()  # gpu OOM error
    GENERATE = auto()  # some error happened while running a text generation request
    TIMEOUT = auto()  # grpc deadline exceeded
    UNKNOWN = auto()


class ServiceMetrics:
    def __init__(self):
        # Tokenization API metrics
        self.tgi_tokenize_request_tokens = Histogram(
            "tgi_tokenize_request_tokens",
            documentation="Histogram of tokenized tokens per tokenize request",
            buckets=[1 << x for x in range(6, 20)],
        )
        self.tgi_tokenize_request_input_count = Counter(
            "tgi_tokenize_request_input_count",
            documentation="Count of tokenize request inputs (batch of n counts as n)",
        )

        # Generate API metrics
        self.tgi_request_input_count = Counter(
            "tgi_request_input_count",
            documentation="Count of generate request inputs (batch of n counts as n)",
        )
        # err = validation|cancelled|conc_limit
        self.tgi_request_failure = Counter(
            "tgi_request_failure",
            labelnames=["err"],
            documentation="Count of failed requests, segmented by error type",
        )
        # The queue duration info from the vllm engine is only known at
        # response time
        self.tgi_request_queue_duration = Histogram(
            "tgi_request_queue_duration",
            documentation="Request time spent in queue (in seconds)",
            buckets=_duration_buckets,
        )
        # Total response time from server
        self.tgi_request_duration = Histogram(
            "tgi_request_duration",
            documentation="End-to-end generate request duration (in seconds)",
            buckets=_duration_buckets,
        )

    def count_tokenization_request(self, request: BatchedTokenizeRequest) -> None:
        self.tgi_tokenize_request_input_count.inc(len(request.requests))

    def observe_tokenization_response(self, response: BatchedTokenizeResponse) -> None:
        for tokenize_response in response.responses:
            self.tgi_tokenize_request_tokens.observe(tokenize_response.token_count)

    def count_generate_request(self, num_requests: int = 1) -> None:
        self.tgi_request_input_count.inc(num_requests)

    def observe_queue_time(self, engine_output: RequestOutput) -> None:
        assert engine_output.metrics

        self.tgi_request_queue_duration.observe(engine_output.metrics.time_in_queue)

    def count_request_failure(self, reason: FailureReasonLabel) -> None:
        self.tgi_request_failure.labels(err=reason).inc(1)

    def observe_generation_success(self, start_time: float) -> None:
        duration = time.time() - start_time
        self.tgi_request_duration.observe(duration)


class TGISStatLogger(StatLoggerBase):
    """Wraps the vLLM StatLogger to report TGIS metric names for compatibility."""

    def __init__(self, vllm_stat_logger: StatLoggerBase, max_sequence_len: int):
        # Not calling super-init because we're wrapping and delegating to
        # vllm_stat_logger
        self._vllm_stat_logger = vllm_stat_logger

        self.tgi_queue_size = Gauge(
            "tgi_queue_size", documentation="Current number of queued requests"
        )
        self.tgi_batch_current_size = Gauge(
            "tgi_batch_current_size", documentation="Current batch size"
        )
        # method = prefill|next_token
        self.tgi_batch_inference_duration = Histogram(
            "tgi_batch_inference_duration",
            labelnames=["method"],
            documentation="Time taken for each forward-pass iteration (in seconds)",
            buckets=_duration_buckets,
        )

        sequence_len_buckets = [max_sequence_len / 64.0 * (x + 1) for x in range(64)]
        self.tgi_request_input_length = Histogram(
            "tgi_request_input_length",
            documentation="Request input length in tokens",
            buckets=sequence_len_buckets,
        )
        self.tgi_request_generated_tokens = Histogram(
            "tgi_request_generated_tokens",
            documentation="Number of tokens generated for request",
            buckets=sequence_len_buckets,
        )

    def info(self, type_: str, obj: object) -> None:
        self._vllm_stat_logger.info(type_, obj)

    def log(self, stats: Stats) -> None:
        # First, log the vLLM stats
        self._vllm_stat_logger.log(stats)

        # Then log TGIS specific ones
        self.tgi_queue_size.set(stats.num_waiting_sys + stats.num_swapped_sys)
        self.tgi_batch_current_size.set(stats.num_running_sys)

        for ttft in stats.time_to_first_tokens_iter:
            self.tgi_batch_inference_duration.labels(method="prefill").observe(ttft)
        for tpot in stats.time_per_output_tokens_iter:
            self.tgi_batch_inference_duration.labels(method="next_token").observe(tpot)

        for input_len in stats.num_prompt_tokens_requests:
            self.tgi_request_input_length.observe(input_len)
        for output_len in stats.num_generation_tokens_requests:
            self.tgi_request_generated_tokens.observe(output_len)
