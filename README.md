# vllm-tgis-adapter

vLLM adapter for a TGIS-compatible grpc server.

[![PyPi](https://img.shields.io/pypi/v/vllm-tgis-adapter?label=pip)](https://pypi.org/project/vllm-tgis-adapter)
[![Tests](https://github.com/dtrifiro/vllm-tgis-adapter/actions/workflows/tests.yaml/badge.svg)](https://github.com/dtrifiro/vllm-tgis-adapter/actions/workflows/tests.yaml)
[![Docker Image Build](https://github.com/dtrifiro/vllm-tgis-adapter/actions/workflows/image.yml/badge.svg)](https://github.com/dtrifiro/vllm-tgis-adapter/actions/workflows/image.yml)
[![quay.io/dtrifiro/vllm-tgis](https://img.shields.io/badge/quay.io-dtrifiro/vllm--tgis-darkred)](https://quay.io/repository/dtrifiro/vllm-tgis?tab=tags)

## Install

vllm-tgis-adapter is available on [PyPi](https://pypi.org/project/vllm-tgis-adapter)

```bash
pip install vllm-tgis-adapter
python -m vllm_tgis_adapter
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

Image available at [quay.io/dtrifiro/vllm-tgis](https://quay.io/dtrifiro/vllm-tgis?tab=tags)

```bash
docker pull quay.io/dtrifiro/vllm-tgis
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
