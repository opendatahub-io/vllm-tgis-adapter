import asyncio
from collections.abc import Iterable


def check_for_failed_tasks(tasks: Iterable[asyncio.Task]) -> None:
    """Check a sequence of tasks exceptions and raise the exception."""
    for task in tasks:
        try:
            exc = task.exception()
        except asyncio.InvalidStateError:
            # no exception is set
            continue

        if not exc:
            continue

        name = task.get_name()
        coro_name = task.get_coro().__name__

        raise RuntimeError(f"task={name} ({coro_name}) exception={exc!s}") from exc


def write_termination_log(msg: str, file: str = "/dev/termination-log") -> None:
    """Write to the termination logfile."""
    # From https://github.com/IBM/text-generation-inference/blob/9388f02d222c0dab695bea1fb595cacdf08d5467/server/text_generation_server/utils/termination.py#L4
    try:
        with open(file, "w") as termination_file:
            termination_file.write(f"{msg}\n")
    except Exception:
        # Ignore any errors writing to the termination logfile.
        # Users can fall back to the stdout logs, and we don't want to pollute
        # those with an error here.
        from .logging import init_logger

        logger = init_logger("vllm-tgis-adapter")
        logger.exception("Unable to write termination logs to %s", file)
