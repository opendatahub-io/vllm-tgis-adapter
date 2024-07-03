# vllm-tgis-adapter

vLLM adapter for a TGIS-compatible grpc server.

[![PyPi](https://img.shields.io/pypi/v/vllm-tgis-adapter?label=pip)](https://pypi.org/project/vllm-tgis-adapter)
[![Tests](https://github.com/opendatahub-io/vllm-tgis-adapter/actions/workflows/tests.yaml/badge.svg)](https://github.com/opendatahub-io/vllm-tgis-adapter/actions/workflows/tests.yaml)
[![quay.io/opendatahub/vllm](https://img.shields.io/badge/quay.io-opendatahub/vllm--tgis-darkred)](https://quay.io/repository/opendatahub/vllm?tab=tags)
[![codecov](https://codecov.io/github/opendatahub-io/vllm-tgis-adapter/branch/main/graph/badge.svg?token=1HVSOP6N0J)](https://codecov.io/github/opendatahub-io/vllm-tgis-adapter)

## Install

vllm-tgis-adapter is available on [PyPi](https://pypi.org/project/vllm-tgis-adapter)

```bash
pip install vllm-tgis-adapter
python -m vllm_tgis_adapter
```

### HealthCheck CLI

Installing the adapter also install a grpc healthcheck cli that can be used to monitor the status of the grpc server:

```console
$ grpc_healtheck
health check...status: SERVING
```

See usage with

```bash
grpc_healthcheck --help
```

## Build

```bash
python -m build
pip install dist/*whl
python -m vllm_tgis_adapter
```

## Inference

This will start serving a grpc server on port 8033. This can be queried with grpcurl:

```bash
bash examples/inference.sh
```

### Docker

Image available at [quay.io/opendatahub/vllm](https://quay.io/opendatahub/vllm?tab=tags), built from [opendatahub-io/vllm](https://github.com/opendatahub-io/vllm)'s [Dockerfile.ubi](https://github.com/opendatahub-io/vllm/tree/main/Dockerfile.ubi)

```bash
docker pull quay.io/opendatahub/vllm
```

### Inference

See [examples](/examples)

## Contributing

Set up [`pre-commit`](https://pre-commit.com) for linting/style/misc fixes:

```bash
pip install pre-commit
pre-commit install
# to run on all files
pre-commit run --all-files
```

This project uses [`nox`](https://github.com/wntrblm/nox) to manage test automation:

```bash
pip install nox
nox --list  # list available sessions
nox -s tests-3.10 # run tests session for a specific python version
nox -s build-3.11 # build the wheel package
nox -s lint-3.11 -- --mypy # run linting with type checks
```

### Testing without a GPU

The standard vllm built requires an Nvidia GPU. When this is not available, it is possible to compile `vllm` from source with CPU support:

```bash

env \
    VLLM_CPU_DISABLE_AVX512=true VLLM_TARGET_DEVICE=cpu \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    pip install git+https://github.com/vllm-project/vllm
```

making it possible to run the tests on most hardware. Please note that the `pip` extra index url is required in order to install the torch CPU version.
