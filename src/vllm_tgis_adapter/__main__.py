import asyncio

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.usage.usage_lib import UsageContext

from .grpc import start_grpc_server
from .logging import init_logger
from .tgis_utils.args import EnvVarArgumentParser, add_tgis_args, postprocess_tgis_args

TIMEOUT_KEEP_ALIVE = 5  # seconds


logger = init_logger(__name__)


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

    try:
        while True:
            await asyncio.sleep(60)
    except asyncio.CancelledError:
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
