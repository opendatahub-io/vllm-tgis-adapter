from __future__ import annotations

import argparse
import sys
import warnings

import grpc
import grpc.experimental  # this is required for grpc_health
from grpc_health.v1.health_pb2 import HealthCheckRequest
from grpc_health.v1.health_pb2_grpc import Health

from vllm_tgis_adapter.grpc.grpc_server import TextGenerationService

warnings.simplefilter(
    action="ignore", category=grpc.experimental.ExperimentalApiWarning
)


def health_check(
    *,
    server_url: str = "localhost:8033",
    service: str | None = None,
    insecure: bool = True,
    timeout: float = 1,
) -> bool:
    print("health check...", end="")
    request = HealthCheckRequest(service=service)
    try:
        response = Health.Check(
            request=request,
            target=server_url,
            timeout=timeout,
            insecure=insecure,
        )
    except grpc.RpcError as e:
        print(f"Health.Check failed: code={e.code()}, details={e.details()}")
        return False

    print(str(response).strip())
    return response.status == response.SERVING


def cli() -> None:
    args = parse_args()
    if not health_check(
        server_url=args.server_url,
        service=args.service_name,
        insecure=args.insecure,
        timeout=args.timeout,
    ):
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--insecure",
        dest="insecure",
        action="store_true",
        help="Use an insecure connection",
    )
    group.add_argument(
        "--secure",
        dest="secure",
        action="store_true",
        help="Use a secure connection",
    )
    group.set_defaults(insecure=True, secure=False)
    parser.add_argument(
        "--server-url",
        type=str,
        help="grpc server url (`host:port`)",
        default="localhost:8033",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Timeout for healthcheck request",
        default=1,
    )
    parser.add_argument(
        "--service-name",
        type=str,
        help="Name of the service to check",
        required=False,
        default=TextGenerationService.SERVICE_NAME,
    )

    return parser.parse_args()


if __name__ == "__main__":
    cli()
