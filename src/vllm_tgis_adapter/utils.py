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

        raise RuntimeError(f"task={name} ({coro_name})") from exc
