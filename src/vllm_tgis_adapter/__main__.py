from __future__ import annotations

import asyncio
import importlib
import inspect
import re
import signal
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import TYPE_CHECKING

import fastapi
import vllm
from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount
from uvicorn import Config as UvicornConfig
from uvicorn import Server as UvicornServer
from vllm import envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (  # noqa: TCH002 # pydantic needs to access these annotations
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    DetokenizeRequest,
    DetokenizeResponse,
    EmbeddingRequest,
    ErrorResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser

from .grpc import run_grpc_server
from .logging import init_logger
from .tgis_utils.args import EnvVarArgumentParser, add_tgis_args, postprocess_tgis_args

if TYPE_CHECKING:
    import argparse
    from collections.abc import AsyncGenerator

    from vllm.config import ModelConfig


try:
    from vllm.entrypoints.openai.serving_tokenization import (
        OpenAIServingTokenization,  # noqa: TCH002
    )
except ImportError:  #  vllm<=0.5.2
    has_tokenization = False
else:
    has_tokenization = True

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
openai_serving_embedding: OpenAIServingEmbedding
if has_tokenization:
    openai_serving_tokenization: OpenAIServingTokenization

logger = init_logger(__name__)

_running_tasks: set[asyncio.Task] = set()


router = APIRouter()


def mount_metrics(app: fastapi.FastAPI) -> None:
    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app())
    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


@router.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


if has_tokenization:
    assert has_tokenization

    @router.post("/tokenize")
    async def tokenize(request: TokenizeRequest) -> JSONResponse:
        generator = await openai_serving_tokenization.create_tokenize(request)  # noqa: F821
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(),
                status_code=generator.code,
            )
        assert isinstance(generator, TokenizeResponse)
        return JSONResponse(content=generator.model_dump())

    @router.post("/detokenize")
    async def detokenize(request: DetokenizeRequest) -> JSONResponse:
        generator = await openai_serving_tokenization.create_detokenize(request)  # noqa: F821
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(),
                status_code=generator.code,
            )

        assert isinstance(generator, DetokenizeResponse)
        return JSONResponse(content=generator.model_dump())


@router.get("/v1/models")
async def show_available_models() -> JSONResponse:
    models = await openai_serving_completion.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.get("/version")
async def show_version() -> JSONResponse:
    ver = {"version": vllm.__version__, "commit": vllm.__commit__}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: fastapi.Request,
) -> JSONResponse:
    generator = await openai_serving_chat.create_chat_completion(
        request,
        raw_request,
    )
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")

    assert isinstance(generator, ChatCompletionResponse)
    return JSONResponse(content=generator.model_dump())


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: fastapi.Request):  # noqa: ANN201
    generator = await openai_serving_completion.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(),
            status_code=generator.code,
        )
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    return JSONResponse(content=generator.model_dump())


@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: fastapi.Request):  # noqa: ANN201
    generator = await openai_serving_embedding.create_embedding(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    return JSONResponse(content=generator.model_dump())


def build_app(  # noqa: C901
    engine: AsyncLLMEngine, args: argparse.Namespace
) -> fastapi.FastAPI:
    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI) -> AsyncGenerator:  # noqa: ARG001
        async def _force_log():  # noqa: ANN202
            while True:
                await asyncio.sleep(10)
                await engine.do_log_stats()

        if not args.disable_log_stats:
            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)

        yield

    app = fastapi.FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):  # noqa: ANN001, ANN202
        err = openai_serving_chat.create_error_response(message=str(exc))
        return JSONResponse(
            err.model_dump(),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: fastapi.Request, call_next):  # noqa: ANN001, ANN202
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(
                    content={"error": "Unauthorized"},
                    status_code=401,
                )
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. " f"Must be a function or a class."
            )

    return app


async def run_http_server(
    engine: AsyncLLMEngine,
    args: argparse.Namespace,
    model_config: ModelConfig,
) -> None:
    app = build_app(engine, args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    global openai_serving_chat  # noqa: PLW0603
    global openai_serving_completion  # noqa: PLW0603
    global openai_serving_embedding  # noqa: PLW0603

    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        args.response_role,
        args.lora_modules,
        args.chat_template,
    )

    openai_serving_completion = OpenAIServingCompletion(
        engine,
        model_config,
        served_model_names,
        args.lora_modules,
        prompt_adapters=args.prompt_adapters,
    )
    openai_serving_embedding = OpenAIServingEmbedding(
        engine, model_config, served_model_names
    )
    app.root_path = args.root_path
    config = UvicornConfig(
        app,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )

    server = UvicornServer(config)
    try:
        await server.serve()
    except asyncio.CancelledError:
        print("Gracefully stopping http server")  # noqa: T201
        await server.shutdown()


if __name__ == "__main__":
    parser = FlexibleArgumentParser("vLLM TGIS GRPC + OpenAI Rest api server")
    # convert to our custom env var arg parser
    parser = EnvVarArgumentParser(parser=make_arg_parser(parser))
    parser = add_tgis_args(parser)
    args = postprocess_tgis_args(parser.parse_args())
    assert args is not None

    version_info = (
        f"{vllm.__version__}" + vllm.__commit__
        if vllm.__commit__ != "COMMIT_HASH_PLACEHOLDER"
        else "unknown"
    )
    logger.info("vLLM version %s", version_info)
    logger.info("args: %s", args)

    engine_args = AsyncEngineArgs.from_cli_args(args)

    if hasattr(engine_args, "image_input_type") and (  # vllm <= 0.5.0.post1
        engine_args.image_input_type is not None
        and engine_args.image_input_type.upper() != "PIXEL_VALUES"
    ):
        # Enforce pixel values as image input type for vision language models
        # when serving with API server
        raise ValueError(
            f"Invalid image_input_type: {engine_args.image_input_type}. "
            "Only --image-input-type 'pixel_values' is supported for serving "
            "vision language models with the vLLM API server."
        )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args,  # type: ignore[arg-type]
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    event_loop: asyncio.AbstractEventLoop | None
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    if event_loop is None:
        event_loop = asyncio.new_event_loop()

    async def run() -> None:
        loop = asyncio.get_running_loop()

        http_server_task = loop.create_task(run_http_server(engine, args, model_config))
        grpc_server_task = loop.create_task(
            run_grpc_server(
                engine, args, disable_log_stats=engine_args.disable_log_stats
            )
        )

        def signal_handler() -> None:
            # prevents the uvicorn signal handler to exit early
            grpc_server_task.cancel()
            http_server_task.cancel()

        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)

        await asyncio.gather(grpc_server_task, http_server_task)

    event_loop.run_until_complete(run())
