from __future__ import annotations

import asyncio
import inspect
import os
import time
import uuid
from collections.abc import Callable, Coroutine
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import grpc
from grpc import StatusCode, aio
from grpc._cython.cygrpc import AbortError
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.async_llm_engine import _AsyncLLMEngine
from vllm.entrypoints.openai.serving_completion import merge_async_iterators
from vllm.inputs import TextTokensPrompt
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)

from vllm_tgis_adapter.logging import init_logger
from vllm_tgis_adapter.tgis_utils import logs
from vllm_tgis_adapter.tgis_utils.guided_decoding import (
    get_outlines_guided_decoding_logits_processor,
)
from vllm_tgis_adapter.tgis_utils.logits_processors import (
    ExpDecayLengthPenaltyWarper,
    TypicalLogitsWarperWrapper,
)
from vllm_tgis_adapter.tgis_utils.metrics import (
    FailureReasonLabel,
    ServiceMetrics,
    TGISStatLogger,
)

from .pb import generation_pb2_grpc
from .pb.generation_pb2 import DESCRIPTOR as _GENERATION_DESCRIPTOR
from .pb.generation_pb2 import (
    BatchedGenerationResponse,
    BatchedTokenizeResponse,
    DecodingMethod,
    GenerationResponse,
    ModelInfoResponse,
    StopReason,
    TokenInfo,
    TokenizeResponse,
)
from .validation import validate_input, validate_params

try:
    from .adapters import AdapterStore, validate_adapters
except ImportError:
    adapters_available = False
else:
    adapters_available = True


if TYPE_CHECKING:
    import argparse
    from collections.abc import AsyncIterator, MutableSequence

    from grpc.aio import ServicerContext
    from transformers import PreTrainedTokenizer
    from vllm import CompletionOutput, RequestOutput
    from vllm.config import ModelConfig
    from vllm.lora.request import LoRARequest
    from vllm.sequence import Logprob

    from .pb.generation_pb2 import (
        BatchedGenerationRequest,
        BatchedTokenizeRequest,
        ModelInfoRequest,
        Parameters,
        ResponseOptions,
        SingleGenerationRequest,
    )

    try:
        from .adapters import PromptAdapterRequest
    except ImportError:
        pass

_T = TypeVar("_T")
_F = TypeVar("_F", Callable, Coroutine)

logger = init_logger(__name__)
service_metrics = ServiceMetrics()

ADD_SPECIAL_TOKENS = os.getenv("ADD_SPECIAL_TOKENS")
if ADD_SPECIAL_TOKENS is not None:
    ADD_SPECIAL_TOKENS = ADD_SPECIAL_TOKENS.lower() not in (0, "false")


def with_default(value: _T, default: _T) -> _T:
    return value if value else default


async def _handle_exception(
    e: Exception,
    func: Callable,
    *args: tuple[Any],
    **kwargs: dict[str, Any],
) -> None:
    context: ServicerContext = kwargs.get("context", None) or args[-1]
    is_generate_fn = "generate" in func.__name__.lower()

    # First just try to replicate the TGIS-style log messages
    # for generate_* rpcs
    if is_generate_fn:
        if isinstance(e, AbortError):
            # For things that we've already aborted, the relevant error
            # string is already in the grpc context.
            error_message = context.details()
        else:
            error_message = str(e)
        request = kwargs.get("request", None) or args[-2]
        logs.log_error(
            request=request,
            exception_str=error_message,
            logger=logger,
        )

    # AbortErrors likely correspond to things we've already explicitly handled,
    # So we only add special handling for other types of errors
    if not isinstance(e, AbortError):
        from torch.cuda import OutOfMemoryError

        if isinstance(e, OutOfMemoryError):
            logger.exception("%s caused GPU OOM error", func.__name__)
            service_metrics.count_request_failure(FailureReasonLabel.OOM)
            await context.abort(StatusCode.RESOURCE_EXHAUSTED, str(e))
        elif is_generate_fn:
            service_metrics.count_request_failure(FailureReasonLabel.GENERATE)
        else:
            service_metrics.count_request_failure(FailureReasonLabel.UNKNOWN)
        logger.exception("%s failed", func.__name__)
    raise e


def log_rpc_handler_errors(func: _F) -> _F:
    if inspect.isasyncgenfunction(func):

        async def func_with_log(*args, **kwargs):  # noqa: ANN002,ANN003,ANN202
            try:
                async for val in func(*args, **kwargs):
                    yield val
            except Exception as e:  # noqa: BLE001
                await _handle_exception(e, func, *args, **kwargs)
    else:

        async def func_with_log(*args, **kwargs):  # noqa: ANN002,ANN003,ANN202
            try:
                return await func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                await _handle_exception(e, func, *args, **kwargs)

    return func_with_log


class TextGenerationService(generation_pb2_grpc.GenerationServiceServicer):
    SERVICE_NAME = _GENERATION_DESCRIPTOR.services_by_name[
        "GenerationService"
    ].full_name

    def __init__(
        self,
        engine: AsyncLLMEngine,
        args: argparse.Namespace,
        health_servicer: health.HealthServicer,
    ):
        self.engine: AsyncLLMEngine = engine

        # This is set in post_init()
        self.config: ModelConfig | None = None

        self.max_max_new_tokens = args.max_new_tokens
        self.skip_special_tokens = not args.output_special_tokens
        self.default_include_stop_seqs = args.default_include_stop_seqs

        # Backwards compatibility for TGIS: PREFIX_STORE_PATH
        adapter_cache_path = args.adapter_cache or args.prefix_store_path
        self.adapter_store = (
            AdapterStore(cache_path=adapter_cache_path, adapters={})
            if adapter_cache_path
            else None
        )
        self.health_servicer = health_servicer

    async def post_init(self) -> None:
        self.config = await self.engine.get_model_config()

        # Swap in the special TGIS stats logger
        tgis_stats_logger = TGISStatLogger(
            vllm_stat_logger=self.engine.engine.stat_loggers["prometheus"],
            max_sequence_len=self.config.max_model_len,
        )
        self.engine.engine.stat_loggers["prometheus"] = tgis_stats_logger

        self.health_servicer.set(
            self.SERVICE_NAME,
            health_pb2.HealthCheckResponse.SERVING,
        )

    @log_rpc_handler_errors
    async def Generate(
        self,
        request: BatchedGenerationRequest,
        context: ServicerContext,
    ) -> BatchedGenerationResponse:
        start_time = time.time()
        service_metrics.count_generate_request(len(request.requests))
        request_id = self.request_id(context)
        adapter_kwargs = (
            await self._validate_adapters(request, context)
            if adapters_available
            else {}
        )
        tokenizer = await self._get_tokenizer(adapter_kwargs)

        sampling_params, deadline = await self._validate_and_convert_params(
            request.params, tokenizer, context
        )
        truncate_input_tokens = with_default(request.params.truncate_input_tokens, None)
        request_count = len(request.requests)

        generators = []
        max_is_token_limit = [False] * request_count

        for i, req in enumerate(request.requests):
            input_ids, max_is_token_limit[i] = await self._validate_prompt_and_tokenize(
                sampling_params, truncate_input_tokens, req.text, tokenizer, context
            )

            inputs = TextTokensPrompt(
                prompt=req.text,
                prompt_token_ids=input_ids,
            )
            kwargs = {}
            is_tracing_enabled = await self.engine.is_tracing_enabled()
            headers = dict(context.invocation_metadata())
            if is_tracing_enabled:
                kwargs["trace_headers"] = extract_trace_headers(headers)
            elif contains_trace_headers(headers):
                log_tracing_disabled_warning()
            generators.append(
                self.engine.generate(
                    inputs=inputs,
                    sampling_params=sampling_params,
                    request_id=f"{request_id}-{i}",
                    **adapter_kwargs,
                    **kwargs,
                ),
            )

        # TODO handle cancellation
        result_generator: AsyncIterator[tuple[int, RequestOutput]] = (
            merge_async_iterators(*generators)
        )

        resp_options = request.params.response
        responses: list = [None] * request_count
        time_limit_reached = False
        async for i, res in result_generator:
            # if await raw_request.is_disconnected():
            #     # Abort the request if the client disconnects.
            #     await self.engine.abort(f"{request_id}-{i}")
            #     return self.create_error_response("Client disconnected")
            responses[i] = res
            service_metrics.observe_queue_time(res)

            if (
                deadline is not None
                and time.time() >= deadline
                and None not in responses
            ):
                for j in range(request_count):
                    await self.engine.abort(f"{request_id}-{j}")
                time_limit_reached = True
                break

        for i in range(len(responses)):
            res = responses[i]
            response = self._convert_output(
                res.outputs[0],
                resp_options,
                max_is_token_limit=max_is_token_limit[i],
                time_limit_reached=time_limit_reached,
            )
            response = self._convert_input_details(
                res, resp_options, sampling_params, response
            )
            logs.log_response(
                request=request,
                response=response,
                start_time=start_time,
                engine_metrics=res.metrics,
                sub_request_num=i,
                logger=logger,
            )
            service_metrics.observe_generation_success(start_time=start_time)
            responses[i] = response

        return BatchedGenerationResponse(responses=responses)

    @log_rpc_handler_errors
    async def GenerateStream(
        self,
        request: SingleGenerationRequest,
        context: ServicerContext,
    ) -> AsyncIterator[GenerationResponse]:
        start_time = time.time()
        service_metrics.count_generate_request()
        request_id = self.request_id(context)
        adapter_kwargs = (
            await self._validate_adapters(request, context)
            if adapters_available
            else {}
        )
        tokenizer = await self._get_tokenizer(adapter_kwargs)

        sampling_params, deadline = await self._validate_and_convert_params(
            request.params, tokenizer, context
        )
        truncate_input_tokens = with_default(request.params.truncate_input_tokens, None)

        input_ids, max_is_tok_limit = await self._validate_prompt_and_tokenize(
            sampling_params,
            truncate_input_tokens,
            request.request.text,
            tokenizer,
            context,
        )

        inputs = TextTokensPrompt(
            prompt=request.request.text, prompt_token_ids=input_ids
        )

        result_generator = self.engine.generate(
            # prompt is supplied for observability, the text is not
            # re-tokenized when `prompt_token_ids` is supplied
            inputs=inputs,
            sampling_params=sampling_params,
            request_id=request_id,
            **adapter_kwargs,
        )

        resp_options = request.params.response

        first_response = None
        last_response = None
        last_output_length = 0
        last_token_count = 0
        time_limit_reached = False
        full_output = ""
        last_engine_response = None
        # TODO handle cancellation
        async for result in result_generator:
            last_engine_response = result
            if first_response is None:
                service_metrics.observe_queue_time(result)
                first_response = self._convert_input_details(
                    result, resp_options, sampling_params, GenerationResponse()
                )
                last_response = first_response
                yield first_response

            output = result.outputs[0]

            if deadline is not None and time.time() >= deadline:
                await self.engine.abort(request_id)
                time_limit_reached = True

            # Convert output text and token_ids to deltas
            last_response = self._convert_output(
                output,
                resp_options,
                max_is_token_limit=max_is_tok_limit,
                time_limit_reached=time_limit_reached,
                text_start_offset=last_output_length,
                token_start_offset=last_token_count,
            )
            yield last_response

            last_output_length = len(output.text)
            last_token_count = len(output.token_ids)
            # Save full output for logging
            full_output = output.text

            if time_limit_reached:
                break

        # Edit up the first_response for logging purposes only
        if first_response is None:
            # We didn't output anything!
            return

        # Log and record metrics
        assert last_response is not None
        first_response.text = full_output
        first_response.stop_reason = last_response.stop_reason
        first_response.stop_sequence = last_response.stop_sequence
        first_response.generated_token_count = last_response.generated_token_count
        logs.log_response(
            request=request,
            response=first_response,
            start_time=start_time,
            engine_metrics=last_engine_response.metrics
            if last_engine_response
            else None,
            logger=logger,
        )
        service_metrics.observe_generation_success(start_time=start_time)

    def _convert_input_details(
        self,
        result: RequestOutput,
        resp_options: ResponseOptions,
        sampling_params: SamplingParams,
        response: GenerationResponse,
    ) -> GenerationResponse:
        response.input_token_count = len(result.prompt_token_ids)
        if resp_options.input_tokens:
            self._convert_tokens(
                result.prompt_token_ids,
                result.prompt_logprobs,
                include_logprobs=resp_options.token_logprobs,
                include_ranks=resp_options.token_ranks,
                top_n_tokens=resp_options.top_n_tokens,
                token_infos=response.input_tokens,
            )

        if resp_options.input_text:
            response.text = (
                result.prompt if not response.text else result.prompt + response.text
            )

        if sampling_params.seed is not None:
            response.seed = sampling_params.seed
        return response

    def _convert_output(  # noqa: PLR0913
        self,
        output: CompletionOutput,
        resp_options: ResponseOptions,
        *,
        max_is_token_limit: bool,
        time_limit_reached: bool = False,
        text_start_offset: int = 0,
        token_start_offset: int = 0,
    ) -> GenerationResponse:
        stop_reason, stop_sequence = self._convert_reason(
            output,
            max_is_token_limit=max_is_token_limit,
            time_limit_reached=time_limit_reached,
        )
        response = GenerationResponse(
            text=output.text[text_start_offset:],
            generated_token_count=len(output.token_ids),
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
        )

        if resp_options.generated_tokens:
            self._convert_tokens(
                output.token_ids,
                output.logprobs,
                include_logprobs=resp_options.token_logprobs,
                include_ranks=resp_options.token_ranks,
                top_n_tokens=resp_options.top_n_tokens,
                token_infos=response.tokens,
                token_start_offset=token_start_offset,
            )
        return response

    @staticmethod
    def request_id(context: ServicerContext) -> str:  # noqa:  ARG004
        return uuid.uuid4().hex

    async def _validate_and_convert_params(
        self,
        params: Parameters,
        tokenizer: PreTrainedTokenizer,
        context: ServicerContext,
    ) -> tuple[SamplingParams, float | None]:
        """Return (sampling_params, deadline)."""
        # First run TGIS validation to raise errors that match the TGIS api
        try:
            validate_params(params, self.max_max_new_tokens)
        except ValueError as tgis_validation_error:
            service_metrics.count_request_failure(FailureReasonLabel.VALIDATION)
            await context.abort(StatusCode.INVALID_ARGUMENT, str(tgis_validation_error))

        resp_options = params.response
        sampling = params.sampling
        stopping = params.stopping
        decoding = params.decoding
        greedy = params.method == DecodingMethod.GREEDY

        max_new_tokens: int | None = None
        if stopping.max_new_tokens > 0:
            max_new_tokens = stopping.max_new_tokens
        min_new_tokens = max(0, stopping.min_new_tokens)

        logprobs: int | None = (
            1 if (resp_options.token_logprobs or resp_options.token_ranks) else 0
        )
        top_n_tokens = resp_options.top_n_tokens
        if top_n_tokens:
            assert logprobs is not None

            # vLLM will currently return logprobs for n+1 tokens
            # (selected token plus top_n excluding selected)
            logprobs += top_n_tokens
            if greedy and resp_options.token_logprobs:
                logprobs -= 1

        logprobs = with_default(logprobs, None)

        # NEW FUNCTION TO ADD (later)
        # - presence penalty, freq penalty
        # - min_p
        # - beam search (with length_penalty, stop_early, n)

        # TBD (investigate more)
        # - best_of / n
        # - spaces_between_special_tokens
        # - skip_special_tokens (per request)
        # - stop_token_ids

        # to match TGIS, only including typical_p processing
        # when using sampling
        logits_processors = []

        if not greedy and 0.0 < sampling.typical_p < 1.0:
            logits_processors.append(
                TypicalLogitsWarperWrapper(mass=sampling.typical_p)
            )

        if decoding.HasField("length_penalty"):
            length_penalty_tuple = (
                decoding.length_penalty.start_index,
                decoding.length_penalty.decay_factor,
            )

            logits_processors.append(
                ExpDecayLengthPenaltyWarper(
                    length_penalty=length_penalty_tuple,
                    eos_token_id=tokenizer.eos_token_id,
                )
            )

        guided_decode_logit_processor = (
            await get_outlines_guided_decoding_logits_processor(decoding, tokenizer)
        )
        if guided_decode_logit_processor is not None:
            logits_processors.append(guided_decode_logit_processor)

        time_limit_millis = stopping.time_limit_millis
        deadline = (
            time.time() + time_limit_millis / 1000.0 if time_limit_millis > 0 else None
        )

        random_sampling_params: dict[str, Any]
        if greedy:
            random_sampling_params = {"temperature": 0.0}
        else:
            random_sampling_params = {
                "temperature": with_default(sampling.temperature, 1.0),
                "top_k": with_default(sampling.top_k, -1),
                "top_p": with_default(sampling.top_p, 1.0),
                "seed": sampling.seed if sampling.HasField("seed") else None,
            }

        try:
            sampling_params = SamplingParams(
                logprobs=logprobs,
                prompt_logprobs=logprobs if resp_options.input_tokens else None,
                max_tokens=max_new_tokens,
                min_tokens=min_new_tokens,
                repetition_penalty=with_default(decoding.repetition_penalty, 1.0),
                logits_processors=logits_processors,
                stop=with_default(stopping.stop_sequences, None),
                include_stop_str_in_output=stopping.include_stop_sequence
                if stopping.HasField("include_stop_sequence")
                else self.default_include_stop_seqs,
                skip_special_tokens=self.skip_special_tokens,
                **random_sampling_params,
            )
        except ValueError as vllm_validation_error:
            # There may be validation cases caught by vLLM that are not covered
            # by the TGIS api validation
            service_metrics.count_request_failure(FailureReasonLabel.VALIDATION)
            await context.abort(StatusCode.INVALID_ARGUMENT, str(vllm_validation_error))

        return sampling_params, deadline

    async def _validate_adapters(
        self,
        request: SingleGenerationRequest
        | BatchedGenerationRequest
        | TokenizeResponse
        | BatchedTokenizeRequest,
        context: ServicerContext,
    ) -> dict[str, LoRARequest | PromptAdapterRequest]:
        try:
            adapters = await validate_adapters(
                request=request, adapter_store=self.adapter_store
            )
        except ValueError as e:
            service_metrics.count_request_failure(FailureReasonLabel.VALIDATION)
            await context.abort(StatusCode.INVALID_ARGUMENT, str(e))
        return adapters

    async def _get_tokenizer(
        self, adapter_kwargs: dict[str, Any]
    ) -> PreTrainedTokenizer:
        lora_request = adapter_kwargs.get("lora_request")
        try:
            return await self.engine.get_tokenizer(lora_request)
        except TypeError as exc:
            # vllm <= 0.5.2
            if "takes 1 positional argument but 2 were given" not in str(exc):
                raise

            return (
                await self.engine.engine.get_tokenizer_group().get_lora_tokenizer_async(
                    lora_request
                )
            )

    @staticmethod
    def _convert_reason(
        output: CompletionOutput,
        *,
        max_is_token_limit: bool,
        time_limit_reached: bool,
    ) -> tuple[StopReason, str | None]:
        finish_reason = output.finish_reason
        stop_sequence = None
        if finish_reason is None:
            stop_reason = (
                StopReason.TIME_LIMIT if time_limit_reached else StopReason.NOT_FINISHED
            )
        elif finish_reason == "length":
            stop_reason = (
                StopReason.TOKEN_LIMIT if max_is_token_limit else StopReason.MAX_TOKENS
            )
        elif finish_reason == "stop":
            stop_reason = StopReason.STOP_SEQUENCE
            # TODO depends on https://github.com/vllm-project/vllm/pull/2976
            if hasattr(output, "stop_reason"):
                stop_str_or_tok = output.stop_reason
                if stop_str_or_tok is None:
                    stop_reason = StopReason.EOS_TOKEN
                elif isinstance(stop_str_or_tok, str):
                    stop_sequence = stop_str_or_tok
                else:
                    logger.warning(
                        "Unexpected stop_reason type: %s", type(stop_str_or_tok)
                    )
        elif finish_reason == "abort":
            stop_reason = StopReason.CANCELLED
        else:
            logger.warning("Unrecognized finish_reason: %s", finish_reason)
            stop_reason = StopReason.CANCELLED

        return stop_reason, stop_sequence

    def _convert_tokens(  # noqa: PLR0913
        self,
        token_ids: list[int],
        logprobs_list: list[dict[int, Logprob] | None] | None,
        *,
        include_logprobs: bool,
        include_ranks: bool,
        top_n_tokens: int,
        tokenizer: PreTrainedTokenizer,
        token_infos: MutableSequence[TokenInfo],  # OUT
        token_start_offset: int = 0,
    ) -> None:
        if token_start_offset:
            token_ids = token_ids[token_start_offset:]
            if logprobs_list is not None:
                logprobs_list = logprobs_list[token_start_offset:]
        token_texts = tokenizer.convert_ids_to_tokens(token_ids)
        for i, text in enumerate(token_texts):
            token_info = TokenInfo(text=text)
            if logprobs_list is None:
                token_infos.append(token_info)
                continue

            logprobs = logprobs_list[i]
            # Logprobs entry will be None for first prompt token
            if logprobs is None:
                token_infos.append(token_info)
                continue

            if include_logprobs or include_ranks:
                logprob = logprobs[token_ids[i]]
                if include_logprobs:
                    token_info.logprob = logprob.logprob
                if include_ranks:
                    assert logprob.rank is not None

                    token_info.rank = logprob.rank
            if top_n_tokens:
                items = sorted(
                    logprobs.items(),
                    key=lambda item: item[1].logprob,
                    reverse=True,
                )[:top_n_tokens]
                tt_texts = tokenizer.convert_ids_to_tokens([tid for tid, _ in items])

                token_info.top_tokens.extend(
                    TokenInfo.TopToken(
                        text=tt_text,
                        logprob=(logprob.logprob if include_logprobs else None),
                    )
                    for tt_text, (_, logprob) in zip(tt_texts, items)
                )
            token_infos.append(token_info)

    async def _validate_prompt_and_tokenize(  # noqa: PLR0913
        self,
        sampling_params: SamplingParams,
        truncate_input_tokens: int | None,
        prompt: str,
        tokenizer: PreTrainedTokenizer,
        context: ServicerContext,
    ) -> tuple[list[int], bool]:
        assert self.config is not None

        max_model_len = self.config.max_model_len

        # Add special tokens based on env var or else only if the tokenizer
        # does not have a chat template => this is not a chat model
        add_special_tokens = (
            ADD_SPECIAL_TOKENS
            if ADD_SPECIAL_TOKENS is not None
            else not tokenizer.chat_template
        )

        tokenizer_kwargs: dict[str, Any] = {"add_special_tokens": add_special_tokens}
        if truncate_input_tokens is not None:
            tokenizer_kwargs.update(
                {
                    "truncation": True,
                    "max_length": truncate_input_tokens,
                }
            )

        input_ids = tokenizer(prompt, **tokenizer_kwargs).input_ids
        token_num = len(input_ids)

        try:
            validate_input(sampling_params, token_num, max_model_len)
        except ValueError as tgis_validation_error:
            await context.abort(StatusCode.INVALID_ARGUMENT, str(tgis_validation_error))

        max_new_tokens: int | None = sampling_params.max_tokens
        max_is_token_limit = False
        if max_new_tokens is None:
            # TGIS has fixed default (of 20 I think), but I think fine to keep
            # default as effective max here, given paged attention
            sampling_params.max_tokens = min(
                self.max_max_new_tokens, max_model_len - token_num
            )
            max_is_token_limit = True
        elif token_num + max_new_tokens > max_model_len:
            sampling_params.max_tokens = max_model_len - token_num
            max_is_token_limit = True

        return input_ids, max_is_token_limit

    @log_rpc_handler_errors
    async def Tokenize(
        self,
        request: BatchedTokenizeRequest,
        context: ServicerContext,
    ) -> BatchedTokenizeResponse:
        """Handle tokenization requests by tokenizing input texts \

        and returning tokenized results.

        If request.truncate_input_tokens is
        provided, the tokenization will contain the truncated results.

        Args:
        ----
            request (BatchedTokenizeRequest): The tokenization request
                containing texts to be tokenized.
            context (ServicerContext): The context for the RPC call.

        Returns:
        -------
            BatchedTokenizeResponse: The response containing the
                tokenized results.

        """
        # Log the incoming tokenization request for metrics
        service_metrics.count_tokenization_request(request)

        # TODO simplify to only check for lora adapter
        adapter_kwargs = await self._validate_adapters(request, context)
        tokenizer = await self._get_tokenizer(adapter_kwargs)

        responses: list[TokenizeResponse] = []

        # TODO: maybe parallelize, also move convert_ids_to_tokens into the
        # other threads
        for req in request.requests:
            batch_encoding = tokenizer.encode_plus(
                text=req.text, return_offsets_mapping=request.return_offsets
            )

            # Tokenize the input text
            token_ids = batch_encoding.input_ids
            token_count = len(token_ids)

            if 0 < request.truncate_input_tokens < token_count:
                token_count = request.truncate_input_tokens

            # Initialize Tokens from ids
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            offsets = None

            if request.return_offsets:
                offsets = [
                    {"start": start, "end": end}
                    for start, end in batch_encoding.offset_mapping
                    if start is not None and end is not None
                ]
                # Truncate offset list if request.truncate_input_tokens
                offsets = offsets[-token_count:]

            tokens = tokens[-token_count:] if request.return_tokens else None

            responses.append(
                TokenizeResponse(
                    token_count=token_count, tokens=tokens, offsets=offsets
                )
            )

        response = BatchedTokenizeResponse(responses=responses)
        service_metrics.observe_tokenization_response(response)
        return response

    @log_rpc_handler_errors
    async def ModelInfo(
        self,
        request: ModelInfoRequest,  # noqa: ARG002
        context: ServicerContext,  # noqa: ARG002
    ) -> ModelInfoResponse:
        return ModelInfoResponse(
            # vLLM currently only supports decoder models
            model_kind=ModelInfoResponse.ModelKind.DECODER_ONLY,
            max_sequence_length=self.config.max_model_len,
            max_new_tokens=self.max_max_new_tokens,
        )


async def start_grpc_server(
    engine: AsyncLLMEngine, args: argparse.Namespace
) -> aio.Server:
    # Log memory summary after model is loaded
    from torch.cuda import memory_summary

    assert isinstance(engine, AsyncLLMEngine)
    assert isinstance(engine.engine, _AsyncLLMEngine)

    if (device_type := engine.engine.device_config.device.type) == "cuda":
        logger.info(memory_summary(engine.engine.device_config.device))
    else:
        logger.warning("Cannot print device usage for device type: %s", device_type)

    server = aio.server()

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    generation = TextGenerationService(engine, args, health_servicer)
    await generation.post_init()
    generation_pb2_grpc.add_GenerationServiceServicer_to_server(generation, server)

    service_names = (
        health.SERVICE_NAME,
        generation.SERVICE_NAME,
        reflection.SERVICE_NAME,
    )

    reflection.enable_server_reflection(service_names, server)

    host = "0.0.0.0" if args.host is None else args.host  # noqa: S104
    listen_on = f"{host}:{args.grpc_port}"
    ssl_keyfile = args.ssl_keyfile
    ssl_certfile = args.ssl_certfile
    ssl_ca_certs = args.ssl_ca_certs

    if ssl_keyfile and ssl_certfile:
        require_client_auth = False
        try:
            with open(ssl_keyfile, "rb") as f:  # noqa: ASYNC101
                ssl_key = f.read()
        except Exception as e:
            raise ValueError(f"Error reading `ssl_keyfile` file: {ssl_keyfile}") from e
        try:
            with open(ssl_certfile, "rb") as f:  # noqa: ASYNC101
                ssl_cert = f.read()
        except Exception as e:
            raise ValueError(
                f"Error reading `ssl_certfile` file: {ssl_certfile}"
            ) from e
        if ssl_ca_certs:
            require_client_auth = True
            try:
                with open(ssl_ca_certs, "rb") as f:  # noqa: ASYNC101
                    root_certificates = f.read()
            except Exception as e:
                raise ValueError(
                    f"Error reading `ssl_ca_certs` file: {ssl_ca_certs}"
                ) from e
        else:
            root_certificates = None
        server_credentials = grpc.ssl_server_credentials(
            [(ssl_key, ssl_cert)], root_certificates, require_client_auth
        )
        server.add_secure_port(listen_on, server_credentials)
    else:
        server.add_insecure_port(listen_on)

    await server.start()
    logger.info("gRPC Server started at %s", listen_on)

    return server


async def run_grpc_server(
    engine: AsyncLLMEngine,
    args: argparse.Namespace,
    *,
    disable_log_stats: bool,
) -> None:
    assert args is not None

    server = await start_grpc_server(engine, args)

    try:
        while True:
            await asyncio.sleep(10)
            if not disable_log_stats:
                await engine.do_log_stats()
    except asyncio.CancelledError:
        print("Gracefully stopping gRPC server")  # noqa: T201
        await server.stop(30)  # TODO configurable grace
        await server.wait_for_termination()
