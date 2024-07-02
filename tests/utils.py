from __future__ import annotations

import socket
import ssl
import sys
import time
from contextlib import closing
from typing import TYPE_CHECKING, Callable, TypeVar

import grpc

from vllm_tgis_adapter.grpc.pb.generation_pb2 import (
    BatchedGenerationRequest,
    GenerationRequest,
    ModelInfoRequest,
    Parameters,
    SingleGenerationRequest,
    StoppingCriteria,
)
from vllm_tgis_adapter.grpc.pb.generation_pb2_grpc import GenerationServiceStub

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from vllm_tgis_adapter.grpc.pb.generation_pb2 import (
        GenerationResponse,
        ModelInfoResponse,
    )

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


def get_server_certificate(host: str, port: int) -> str:
    """Connect to host:port and get the certificate it presents.

    This is almost the same as `ssl.get_server_certificate`, but
    when opening the TLS socket, `server_hostname` is also provided.

    This retrieves the correct certificate for hosts using name-based
    virtual hosting.
    """
    if sys.version_info >= (3, 10):
        # ssl.get_server_certificate supports TLS SNI only above 3.10
        # https://github.com/python/cpython/pull/16820
        return ssl.get_server_certificate((host, port))

    context = ssl.SSLContext()

    with (
        socket.create_connection((host, port)) as sock,
        context.wrap_socket(sock, server_hostname=host) as ssock,
    ):
        cert_der = ssock.getpeercert(binary_form=True)

    assert cert_der
    return ssl.DER_cert_to_PEM_cert(cert_der)


class GrpcClient:
    def __init__(  # noqa: PLR0913
        self,
        host: str,
        port: int,
        *,
        insecure: bool = False,
        verify: bool | None = None,
        ca_cert: bytes | str | None = None,
        client_cert: bytes | str | None = None,
        client_key: bytes | str | None = None,
    ) -> None:
        self._channel = self._make_channel(
            host,
            port,
            insecure=insecure,
            verify=verify,
            client_key=client_key,
            client_cert=client_cert,
            ca_cert=ca_cert,
        )
        self.generation_service_stub = GenerationServiceStub(channel=self._channel)

    def model_info(self, model_id: str) -> ModelInfoResponse:
        return self.generation_service_stub.ModelInfo(
            request=ModelInfoRequest(model_id=model_id),
        )

    def make_request(
        self,
        text: str | list[str],
        model_id: str | None = None,
        max_new_tokens: int = 10,
    ) -> GenerationResponse | Sequence[GenerationResponse]:
        if single_request := isinstance(text, str):
            text = [text]

        request = BatchedGenerationRequest(
            model_id=model_id,
            requests=[GenerationRequest(text=piece) for piece in text],
            params=Parameters(
                stopping=StoppingCriteria(max_new_tokens=max_new_tokens),
            ),
        )

        response = self.generation_service_stub.Generate(
            request=request,
        )

        if single_request:
            return response.responses[0]

        return response.responses

    def make_request_stream(
        self,
        text: str,
        model_id: str | None = None,
        max_new_tokens: int = 10,
    ) -> Generator[GenerationResponse, None, None]:
        request = SingleGenerationRequest(
            model_id=model_id,
            request=GenerationRequest(text=text),
            params=Parameters(
                stopping=StoppingCriteria(max_new_tokens=max_new_tokens),
            ),
        )

        try:
            yield from self.generation_service_stub.GenerateStream(request=request)
        except grpc._channel._MultiThreadedRendezvous as exc:  # noqa: SLF001
            raise RuntimeError(exc.details()) from exc

    def __enter__(self):  # noqa: D105
        return self

    def __exit__(self, *exc_info):  # noqa: D105
        self._close()
        return False

    def _close(self):
        try:
            if hasattr(self, "_channel") and self._channel:
                self._channel.close()
        except Exception as exc:  # noqa: BLE001
            print(f"Unexpected exception while closing client: {exc}")

    def __del__(self):  # noqa: D105
        self._close()

    def _make_channel(  # noqa: D417,PLR0913
        self,
        host: str,
        port: int,
        *,
        insecure: bool = False,
        verify: bool | None = None,
        ca_cert: bytes | str | None = None,
        client_key: bytes | str | None = None,
        client_cert: bytes | str | None = None,
    ) -> grpc.Channel:
        """Create a grpc channel.

        Args:
        ----
        - host: str
        - port: str
        - (optional) insecure: use a plaintext connection (default=False)
        - (optional) verify: set to False to disable remote host certificate(s)
                     verification. Cannot be used with `plaintext` or with MTLS
        - (optional) ca_cert: certificate authority to use
        - (optional) client_key: client key for mTLS mode
        - (optional) client_cert: client cert for mTLS mode

        """
        if not host.strip():
            raise ValueError("A non empty host name is required")
        if int(port) <= 0:
            raise ValueError("A non zero port number is required")
        if insecure and any(
            (val is not None) for val in (ca_cert, client_key, client_cert)
        ):
            raise ValueError("cannot use insecure with TLS/mTLS certificates")
        if insecure and verify:
            raise ValueError("insecure cannot be used with verify")

        connection = f"{host}:{port}"
        if insecure:
            print("Connecting over an insecure plaintext grpc channel")
            return grpc.insecure_channel(connection)

        client_key_bytes = self._try_load_certificate(client_key)
        client_cert_bytes = self._try_load_certificate(client_cert)
        ca_cert_bytes = self._try_load_certificate(ca_cert)

        credentials_kwargs: dict[str, bytes] = {}
        if ca_cert_bytes and not (any((client_cert_bytes, client_key_bytes))):
            print("Connecting using provided CA certificate for secure channel")
            credentials_kwargs.update(root_certificates=ca_cert_bytes)
        elif client_cert_bytes and client_key_bytes and ca_cert_bytes:
            print("Connecting using mTLS for secure channel")
            credentials_kwargs.update(
                root_certificates=ca_cert_bytes,
                private_key=client_key_bytes,
                certificate_chain=client_cert_bytes,
            )
        elif verify is False:
            print(
                f"insecure mode: trusting remote certificate from {host}:{port}",
            )

            cert = get_server_certificate(host, port).encode()
            credentials_kwargs.update(root_certificates=cert)

        return grpc.secure_channel(
            connection, grpc.ssl_channel_credentials(**credentials_kwargs)
        )

    @staticmethod
    def _try_load_certificate(certificate: bytes | str | None) -> bytes | None:
        """Return contents if the certificate points to a file, return the bytes otherwise."""  # noqa: E501
        if not certificate:
            return None

        if isinstance(certificate, bytes):
            return certificate

        if isinstance(certificate, str):
            with open(certificate, "rb") as secret_file:
                return secret_file.read()
        raise ValueError(
            f"{certificate=} should be a path to a certificate files or bytes"
        )
