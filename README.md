# vllm-tgis-adapter

vLLM adapter for a TGIS-compatible grpc server.

## Get started

```bash
python -m build
pip install dist/*whl
python -m vllm_tgis_adapter
```

This will start serving a grpc server on port 8033. This can be queried with grpcurl:

```bash
bash examples/inference.sh
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
```
