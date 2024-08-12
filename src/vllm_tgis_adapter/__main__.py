from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import FIRST_COMPLETED
from typing import TYPE_CHECKING

import uvloop
import vllm
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser

from .grpc import run_grpc_server
from .http import run_http_server
from .logging import init_logger
from .tgis_utils.args import EnvVarArgumentParser, add_tgis_args, postprocess_tgis_args
from .utils import check_for_failed_tasks

if TYPE_CHECKING:
    import argparse

logger = init_logger("vllm-tgis-adapter")


async def start_servers(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()

    tasks: list[asyncio.Task] = []
    async with build_async_engine_client(args) as engine:
        http_server_task = loop.create_task(
            run_http_server(args, engine),
            name="http_server",
        )
        # The http server task will catch interrupt signals for us
        tasks.append(http_server_task)

        grpc_server_task = loop.create_task(
            run_grpc_server(args, engine),
            name="grpc_server",
        )
        tasks.append(grpc_server_task)

        with contextlib.suppress(asyncio.CancelledError):
            # Both server tasks will exit normally on shutdown, so we await
            # FIRST_COMPLETED to catch either one shutting down.
            await asyncio.wait(tasks, return_when=FIRST_COMPLETED)
        # Once either server shuts down, cancel the other
        for task in tasks:
            task.cancel()

        # Final wait for both servers to finish
        await asyncio.wait(tasks)

        check_for_failed_tasks(tasks)


if __name__ == "__main__":
    parser = FlexibleArgumentParser("vLLM TGIS GRPC + OpenAI REST api server")
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

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.new_event_loop()
    loop.run_until_complete(start_servers(args))
