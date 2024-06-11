import socket
import time
from contextlib import closing
from typing import Callable, TypeVar

_T = TypeVar("_T")


def get_random_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        return port


def wait_until(
    pred: Callable[..., _T],
    timeout: float = 30,
    pause: float = 0.5,
) -> _T:
    start = time.perf_counter()
    exc = None

    while (time.perf_counter() - start) < timeout:
        try:
            value = pred()
        except Exception as e:  # noqa: BLE001
            exc = e
        else:
            return value
        time.sleep(pause)

    raise TimeoutError("timed out waiting") from exc
