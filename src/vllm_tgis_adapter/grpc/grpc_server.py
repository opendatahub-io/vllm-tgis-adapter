from __future__ import annotations

import inspect
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
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.async_llm_engine import _AsyncLLMEngine
from vllm.entrypoints.openai.serving_completion import merge_async_iterators
from vllm.inputs import TextTokensPrompt

from vllm_tgis_adapter.logging import init_logger
from vllm_tgis_adapter.tgis_utils import logs
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

if TYPE_CHECKING:
    import argparse
    from collections.abc import AsyncIterator, MutableSequence

    from grpc.aio import ServicerContext
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
    from vllm import CompletionOutput, RequestOutput
    from vllm.config import ModelConfig
    from vllm.sequence import Logprob
    from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup

    from .pb.generation_pb2 import (
        BatchedGenerationRequest,
        BatchedTokenizeRequest,
        ModelInfoRequest,
        Parameters,
        ResponseOptions,
        SingleGenerationRequest,
    )

_T = TypeVar("_T")
_F = TypeVar("_F", Callable, Coroutine)

logger = init_logger(__name__)
service_metrics = ServiceMetrics()


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
    def __init__(self, engine: AsyncLLMEngine, args: argparse.Namespace):
        self.engine: AsyncLLMEngine = engine

        # These are set in post_init()
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None
        self.config: ModelConfig | None = None

        self.max_max_new_tokens = args.max_new_tokens
        self.skip_special_tokens = not args.output_special_tokens
        self.default_include_stop_seqs = args.default_include_stop_seqs

    @property
    def tokenizer_group(self) -> BaseTokenizerGroup:
        assert hasattr(self.engine.engine, "tokenizer")
        assert self.engine.engine.tokenizer is not None

        return self.engine.engine.tokenizer

    async def post_init(self) -> None:
        self.config = await self.engine.get_model_config()

        self.tokenizer = await self.engine.get_tokenizer()
        assert self.tokenizer is not None

        # Swap in the special TGIS stats logger
        assert hasattr(self.engine.engine, "stat_logger")
        assert self.engine.engine.stat_logger

        vllm_stat_logger = self.engine.engine.stat_logger
        tgis_stats_logger = TGISStatLogger(
            vllm_stat_logger=vllm_stat_logger,
            max_sequence_len=self.config.max_model_len,
        )
        # 🌶️🌶️🌶️ sneaky sneak
        self.engine.engine.stat_logger = tgis_stats_logger

    @log_rpc_handler_errors
    async def Generate(
        self,
        request: BatchedGenerationRequest,
        context: ServicerContext,
    ) -> BatchedGenerationResponse:
        start_time = time.time()
        service_metrics.count_generate_request(len(request.requests))
        request_id = self.request_id(context)
        sampling_params, deadline = await self._validate_and_convert_params(
            request.params, context
        )
        truncate_input_tokens = with_default(request.params.truncate_input_tokens, None)
        request_count = len(request.requests)

        generators = []
        max_is_token_limit = [False] * request_count
        for i, req in enumerate(request.requests):
            input_ids, max_is_token_limit[i] = await self._validate_prompt_and_tokenize(
                sampling_params, truncate_input_tokens, req.text, context
            )

            inputs = TextTokensPrompt(
                prompt=req.text,
                prompt_token_ids=input_ids,
            )
            generators.append(
                self.engine.generate(
                    inputs=inputs,
                    sampling_params=sampling_params,
                    request_id=f"{request_id}-{i}",
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
        sampling_params, deadline = await self._validate_and_convert_params(
            request.params, context
        )
        truncate_input_tokens = with_default(request.params.truncate_input_tokens, None)

        input_ids, max_is_tok_limit = await self._validate_prompt_and_tokenize(
            sampling_params, truncate_input_tokens, request.request.text, context
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
        )

        resp_options = request.params.response

        first = True
        first_response = None
        last_output_length = 0
        last_token_count = 0
        time_limit_reached = False
        full_output = ""
        last_engine_response = None
        # TODO handle cancellation
        async for result in result_generator:
            last_engine_response = result
            if first:
                service_metrics.observe_queue_time(result)
                first_response = self._convert_input_details(
                    result, resp_options, sampling_params, GenerationResponse()
                )
                yield first_response
                first = False

            output = result.outputs[0]

            if deadline is not None and time.time() >= deadline:
                await self.engine.abort(request_id)
                time_limit_reached = True

            # Convert output text and token_ids to deltas
            yield self._convert_output(
                output,
                resp_options,
                max_is_tok_limit,
                time_limit_reached,
                last_output_length,
                last_token_count,
            )
            if time_limit_reached:
                break

            last_output_length = len(output.text)
            last_token_count = len(output.token_ids)
            # Save full output for logging
            full_output = output.text

        # Edit up the first_response for logging purposes only
        if first_response is None:
            # We didn't output anything!
            return
        first_response.text = full_output
        first_response.generated_token_count = last_token_count
        self.log_response(
            request=request,
            response=first_response,
            start_time=start_time,
            engine_metrics=last_engine_response.metrics
            if last_engine_response
            else None,
            logger=logger,
        )

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
        self, params: Parameters, context: ServicerContext
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
            assert self.tokenizer is not None

            logits_processors.append(
                ExpDecayLengthPenaltyWarper(
                    length_penalty=length_penalty_tuple,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            )

        time_limit_millis = stopping.time_limit_millis
        deadline = (
            time.time() + time_limit_millis / 1000.0 if time_limit_millis > 0 else None
        )

        try:
            sampling_params = SamplingParams(
                logprobs=logprobs,
                prompt_logprobs=logprobs if resp_options.input_tokens else None,
                max_tokens=max_new_tokens,
                min_tokens=min_new_tokens,
                temperature=with_default(sampling.temperature, 1.0)
                if not greedy
                else 0.0,
                top_k=with_default(sampling.top_k, -1),
                top_p=with_default(sampling.top_p, 1.0),
                seed=sampling.seed if sampling.HasField("seed") else None,
                repetition_penalty=with_default(decoding.repetition_penalty, 1.0),
                logits_processors=logits_processors,
                stop=with_default(stopping.stop_sequences, None),
                include_stop_str_in_output=stopping.include_stop_sequence
                if stopping.HasField("include_stop_sequence")
                else self.default_include_stop_seqs,
                skip_special_tokens=self.skip_special_tokens,
            )
        except ValueError as vllm_validation_error:
            # There may be validation cases caught by vLLM that are not covered
            # by the TGIS api validation
            service_metrics.count_request_failure(FailureReasonLabel.VALIDATION)
            await context.abort(StatusCode.INVALID_ARGUMENT, str(vllm_validation_error))

        return sampling_params, deadline

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
        token_infos: MutableSequence[TokenInfo],  # OUT
        token_start_offset: int = 0,
    ) -> None:
        assert self.tokenizer

        if token_start_offset:
            token_ids = token_ids[token_start_offset:]
            if logprobs_list is not None:
                logprobs_list = logprobs_list[token_start_offset:]
        # TODO later use get_lora_tokenizer here
        token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
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
                # TODO later use get_lora_tokenizer here
                tt_texts = self.tokenizer.convert_ids_to_tokens(
                    [tid for tid, _ in items]
                )

                token_info.top_tokens.extend(
                    TokenInfo.TopToken(
                        text=tt_text,
                        logprob=(logprob.logprob if include_logprobs else None),
                    )
                    for tt_text, (_, logprob) in zip(tt_texts, items)
                )
            token_infos.append(token_info)

    async def _validate_prompt_and_tokenize(
        self,
        sampling_params: SamplingParams,
        truncate_input_tokens: int | None,
        prompt: str,
        context: ServicerContext,
    ) -> tuple[list[int], bool]:
        assert self.config is not None

        max_model_len = self.config.max_model_len
        # tokenize_kwargs = {"truncation": True,
        #                    "max_length": truncate_input_tokens} \
        #     if truncate_input_tokens is not None else {
        #       "truncation": True, "max_length": max_model_len + 1}
        tokenize_kwargs: dict[str, Any] = {}

        input_ids = await self.tokenizer_group.encode_async(
            prompt,
            **tokenize_kwargs,
        )

        # TODO this is temporary until truncation option is added
        # to the TokenizerGroup encode methods
        if truncate_input_tokens and truncate_input_tokens < len(input_ids):
            input_ids = input_ids[-truncate_input_tokens:]
            if not sampling_params.skip_special_tokens:
                add_bos_token = getattr(self.tokenizer, "add_bos_token", False)
                if add_bos_token:
                    assert self.tokenizer is not None

                    input_ids[0] = self.tokenizer.bos_token_id
        # -----------------------------------------------

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
        self, request: BatchedTokenizeRequest, context: ServicerContext
    ) -> BatchedTokenizeResponse:
        service_metrics.observe_tokenization_request(request)
        # TODO implement these
        if request.return_offsets:
            await context.abort(
                StatusCode.INVALID_ARGUMENT, "return_offsets not yet supported"
            )
        if request.truncate_input_tokens:
            await context.abort(
                StatusCode.INVALID_ARGUMENT, "truncate_input_tokens not yet supported"
            )

        responses: list[TokenizeResponse] = []

        # TODO maybe parallelize, also move convert_ids_to_tokens
        # into the other threads
        for req in request.requests:
            token_ids = await self.tokenizer_group.encode_async(req.text)

            responses.append(
                TokenizeResponse(
                    token_count=len(token_ids),
                    tokens=self.tokenizer.convert_ids_to_tokens(token_ids)
                    if request.return_tokens
                    else None,
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

    logger.info(memory_summary(engine.engine.device_config.device))

    server = aio.server()
    service = TextGenerationService(engine, args)
    await service.post_init()

    generation_pb2_grpc.add_GenerationServiceServicer_to_server(service, server)

    # TODO add reflection

    # SERVICE_NAMES = (
    #     generation_pb2.DESCRIPTOR.services_by_name["GenerationService"]
    #     .full_name,
    #     reflection.SERVICE_NAME,
    # )
    # reflection.enable_server_reflection(SERVICE_NAMES, server)

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
