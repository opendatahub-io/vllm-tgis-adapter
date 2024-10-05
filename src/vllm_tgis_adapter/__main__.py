from __future__ import annotations

import asyncio
import contextlib
import os
import traceback
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
from .utils import check_for_failed_tasks, write_termination_log

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

        runtime_error = None
        with contextlib.suppress(asyncio.CancelledError):
            # Both server tasks will exit normally on shutdown, so we await
            # FIRST_COMPLETED to catch either one shutting down.
            await asyncio.wait(tasks, return_when=FIRST_COMPLETED)
            if engine and engine.errored and not engine.is_running:
                # both servers shut down when an engine error
                # is detected, with task done and exception handled
                # here we just notify of that error and let servers be
                runtime_error = RuntimeError(
                    "AsyncEngineClient error detected, this may be caused by an \
                        unexpected error in serving a request. \
                        Please check the logs for more details."
                )

        failed_task = check_for_failed_tasks(tasks)

        # Once either server shuts down, cancel the other
        for task in tasks:
            task.cancel()

        # Final wait for both servers to finish
        await asyncio.wait(tasks)

        # Raise originally-failed task if applicable
        if failed_task:
            name, coro_name = failed_task.get_name(), failed_task.get_coro().__name__
            exception = failed_task.exception()
            raise RuntimeError(f"Failed task={name} ({coro_name})") from exception

        if runtime_error:
            raise runtime_error


def run_and_catch_termination_cause(
    loop: asyncio.AbstractEventLoop, task: asyncio.Task
) -> None:
    try:
        loop.run_until_complete(task)
    except Exception:
        # Report the first exception as cause of termination
        msg = traceback.format_exc()
        write_termination_log(
            msg, os.getenv("TERMINATION_LOG_DIR", "/dev/termination-log")
        )
        raise


if __name__ == "__main__":
    parser = FlexibleArgumentParser("vLLM TGIS GRPC + OpenAI REST api server")
    # convert to our custom env var arg parser
    parser = EnvVarArgumentParser(parser=make_arg_parser(parser))
    parser = add_tgis_args(parser)
    args = postprocess_tgis_args(parser.parse_args())
    assert args is not None

    logger.info("vLLM version %s", f"{vllm.__version__}")
    logger.info("args: %s", args)

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.new_event_loop()
    task = loop.create_task(start_servers(args))
    run_and_catch_termination_cause(loop, task)
