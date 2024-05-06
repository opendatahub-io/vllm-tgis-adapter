from shlex import split
from subprocess import CalledProcessError, check_call
from textwrap import dedent

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.errors import SetupError


class BuildPyAndGenerateGrpc(build_py):
    """build python module using protoc to prepare generated files."""

    proto_source = "vllm_tgis_adapter/grpc/pb/generation.proto"

    def run(self):
        print(f"Invoking protoc on {self.proto_source}")

        # NOTE: imports in generated files will be broken unless some care is given in
        # how --proto_path, --*_out and .proto paths are given.
        #
        # See https://github.com/grpc/grpc/issues/9575#issuecomment-293934506
        try:
            check_call(
                split(
                    dedent(
                        f"""
                        python -m grpc_tools.protoc \
                            --proto_path=src \
                            --python_out=src/ \
                            --grpc_python_out=src/ \
                            --mypy_out=src/ \
                            {self.proto_source}
                      """,
                    ),
                )
            )
        except CalledProcessError as exc:
            raise SetupError(f"protoc failed, exit code {exc.returncode}") from exc

        super().run()


setup(
    cmdclass={"build_py": BuildPyAndGenerateGrpc},
)
