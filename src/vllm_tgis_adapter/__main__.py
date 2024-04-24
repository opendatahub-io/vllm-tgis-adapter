import argparse

import asyncio

import fastapi
import importlib
import inspect
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

from .grpc import start_grpc_server
from .tgis_utils.args import add_tgis_args, postprocess_tgis_args

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
async_llm_engine: AsyncLLMEngine = None
args: argparse.Namespace | None = None
logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    assert args is not None

    grpc_server = await start_grpc_server(async_llm_engine, args)

    yield

    logger.info("Gracefully stopping gRPC server")
    await grpc_server.stop(30)  # TODO configurable grace
    await grpc_server.wait_for_termination()
    logger.info("gRPC server stopped")


app = fastapi.FastAPI(lifespan=lifespan)

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

if __name__ == "__main__":
    parser = make_arg_parser()
    parser = add_tgis_args(parser)
    args = postprocess_tgis_args(parser.parse_args())
    assert args is not None


    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := os.environ.get("VLLM_API_KEY") or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
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

    logger.info(f"vLLM API server version {vllm.__version__}")
    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )
    openai_serving_chat = OpenAIServingChat(
        engine,
        served_model_names,
        args.response_role,
        args.lora_modules,
        args.chat_template,
    )
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model_names, args.lora_modules
    )
    async_llm_engine = engine

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
