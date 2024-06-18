from __future__ import annotations

import asyncio
import importlib
import inspect
import re
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import TYPE_CHECKING

import fastapi
import uvicorn
import vllm
from fastapi import APIRouter

if TYPE_CHECKING:
    from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount
from vllm import envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        CompletionRequest,
        EmbeddingRequest,
    )
from vllm.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.usage.usage_lib import UsageContext

from .grpc import start_grpc_server
from .logging import init_logger
from .tgis_utils.args import EnvVarArgumentParser, add_tgis_args, postprocess_tgis_args

TIMEOUT_KEEP_ALIVE = 5  # seconds

# ruff: noqa: PLW0603

logger = init_logger(__name__)
engine: AsyncLLMEngine
engine_args: AsyncEngineArgs
openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
openai_serving_embedding: OpenAIServingEmbedding

_running_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> asyncio.Generator[None, asyncio.Any, None]:  # noqa: ARG001
    async def _force_log():  # noqa: ANN202
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield


router = APIRouter()

# Add prometheus asgi middleware to route /metrics requests
route = Mount("/metrics", make_asgi_app())
# Workaround for 307 Redirect for /metrics
route.path_regex = re.compile("^/metrics(?P<path>.*)$")
router.routes.append(route)


@router.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@router.get("/v1/models")
async def show_available_models() -> JSONResponse:
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.get("/version")
async def show_version() -> JSONResponse:
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):  # noqa: ANN201
    generator = await openai_serving_chat.create_chat_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    assert isinstance(generator, ChatCompletionResponse)
    return JSONResponse(content=generator.model_dump())


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):  # noqa: ANN201
    generator = await openai_serving_completion.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    return JSONResponse(content=generator.model_dump())


@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):  # noqa: ANN201
    generator = await openai_serving_embedding.create_embedding(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    return JSONResponse(content=generator.model_dump())


def build_app(args):  # noqa: ANN001, ANN201
    app = fastapi.FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

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
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):  # noqa: ANN001, ANN202
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
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


def run_server(args, llm_engine: AsyncLLMEngine = None) -> None:  # noqa: ANN001
    app = build_app(args)

    logger.info("vLLM API server version %s", vllm.__version__)
    logger.info("args: %s", args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    global engine, engine_args

    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Enforce pixel values as image input type for vision language models
    # when serving with API server
    if (
        engine_args.image_input_type is not None
        and engine_args.image_input_type.upper() != "PIXEL_VALUES"
    ):
        raise ValueError(
            f"Invalid image_input_type: {engine_args.image_input_type}. "
            "Only --image-input-type 'pixel_values' is supported for serving "
            "vision language models with the vLLM API server."
        )

    engine = (
        llm_engine
        if not llm_engine
        else AsyncLLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.OPENAI_API_SERVER
        )
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

    global openai_serving_chat
    global openai_serving_completion
    global openai_serving_embedding

    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        args.response_role,
        args.lora_modules,
        args.chat_template,
    )
    openai_serving_completion = OpenAIServingCompletion(
        engine, model_config, served_model_names, args.lora_modules
    )
    openai_serving_embedding = OpenAIServingEmbedding(
        engine, model_config, served_model_names
    )
    app.root_path = args.root_path
    uvicorn.run(
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


async def grpc_server(
    async_llm_engine: AsyncLLMEngine,
    *,
    disable_log_stats: bool,
) -> None:
    async def _force_log() -> None:
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not disable_log_stats:
        asyncio.create_task(_force_log())  # noqa: RUF006

    assert args is not None

    server = await start_grpc_server(async_llm_engine, args)

    yield

    logger.info("Gracefully stopping gRPC server")
    await server.stop(30)  # TODO configurable grace
    await server.wait_for_termination()
    logger.info("gRPC server stopped")


if __name__ == "__main__":
    # convert to our custom env var arg parser
    parser = EnvVarArgumentParser(parser=make_arg_parser())
    parser = add_tgis_args(parser)
    args = postprocess_tgis_args(parser.parse_args())
    assert args is not None

    logger.info("vLLM API grpc server version %s", vllm.__version__)
    logger.info("args: %s", args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args,  # type: ignore[arg-type]
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    asyncio.run(
        grpc_server(
            engine,
            disable_log_stats=engine_args.disable_log_stats,
        ),
    )

    run_server(args, llm_engine=engine)
