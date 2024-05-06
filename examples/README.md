# tgis example

## Introduction

This repo includes proto files from [text-generation-inference](https://github.com/opendatahub-io/text-generation-inference).

```bash
git clone https://github.com/opendatahub-io/text-generation-inference
(cd text-generation-inference && git checkout a57bdf1)
# optional: re-generate python modules:
python -m grpc_tools.protoc -I proto --python_out=. --pyi_out=. --grpc_python_out=. text-generation-inference/proto/generation.proto
```

## Inference

### Using bash+grpcurl

```bash
export GRPC_HOSTNAME=<host name>
export GRPC_PORT=<port>
bash grpcurl_request.sh
```

**Notes:**

- this does not validate TLS certs (uses `grpcurl -insecure`).
- If not using TLS, replace `-insecure` with `-plaintext`

### python

```bash
python -m venv .venv
source .venv/bin/activate

pip install grpcio-tools
python inference.py # runs inference
```

See Introduction section for help regarding python stubs generation.

### Notebooks

untested
