# vllm-tgis-adapter

vLLM adapter for a TGIS-compatible grpc server.

## Get started

```bash
python -m build
pip instal dist/*whl
python -m vllm_tgis_adapter
```

This will start serving a grpc server on port 8033. This can be queried with grpcurl:

```bash
bash examples/inference.sh
```
