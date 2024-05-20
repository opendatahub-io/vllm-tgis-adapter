ARG BASE_UBI_IMAGE_TAG=9.3-1612
ARG PYTHON_VERSION=3.11

FROM registry.access.redhat.com/ubi9/ubi-minimal:${BASE_UBI_IMAGE_TAG} as base

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN microdnf install -y \
    python3.11-pip python3.11-wheel \
    && microdnf clean all

WORKDIR /workspace


FROM base as python-base

ARG PYTHON_VERSION

ENV VIRTUAL_ENV=/opt/vllm
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# hadolint ignore=DL3041
RUN microdnf install -y \
    git \
    python${PYTHON_VERSION}-devel python${PYTHON_VERSION}-pip python${PYTHON_VERSION}-wheel && \
    python${PYTHON_VERSION} -m venv $VIRTUAL_ENV && microdnf clean all

# hadolint ignore=DL3042,DL3013
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip

# TODO: add flash attention build

FROM python-base AS build
ARG PYTHON_VERSION

ENV VIRTUAL_ENV=/opt/vllm
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=python-base /opt/vllm /opt/vllm

# hadolint ignore=DL3042
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nox==2023.4.22 # TODO: setup renovate

COPY README.md .

COPY pyproject.toml .
COPY noxfile.py .

COPY src src

# setuptools scm requires the git directory to infer version from git tags, so we bind-mount the git dir when building
RUN --mount=type=bind,source=.git,target=.git \
    --mount=type=cache,target=.nox \
    --mount=type=cache,target=/root/.cache/pip \
    nox -s build-${PYTHON_VERSION}


FROM base AS deploy
ARG flash_attn_version=2.5.8
ARG cuda_version_flashattn=122
ARG torch_version=2.1

WORKDIR /workspace

COPY --from=python-base /opt/vllm /opt/vllm

ENV VIRTUAL_ENV=/opt/vllm
ENV PATH=$VIRTUAL_ENV/bin/:$PATH

# Triton needs a CC compiler
# hadolint ignore=DL3041
RUN microdnf install -y gcc \
    && microdnf clean all

ENV FLASH_ATTN_VERSION=${flash_attn_version}
ENV CUDA_VERSION_FLASHATTN=${cuda_version_flashattn}
ENV CUDA_VERSION="12.0.0"
ENV TORCH_VERSION=${torch_version}


# make sure that the python version in the flash-attn wheel install below matches ${PYTHON_VERSION}
# hadolint ignore=DL3042
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=build,src=/workspace/dist/,target=/workspace/dist/ \
    pip install \
        https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu${CUDA_VERSION_FLASHATTN}torch${TORCH_VERSION}cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
        dist/*whl

# vllm requires a specific nccl version built from source distribution
# See https://github.com/NVIDIA/nccl/issues/1234
RUN pip install \
        -v \
        --force-reinstall \
        --no-binary="all" \
        --no-cache-dir \
        "vllm-nccl-cu12==2.18.1.0.4.0" && \
    mv /root/.config/vllm/nccl/cu12/libnccl.so.2.18.1 /opt/vllm/lib/ && \
    chmod 0755 /opt/vllm/lib/libnccl.so.2.18.1

ENV HF_HUB_OFFLINE=1 \
    PORT=8000 \
    GRPC_PORT=8033 \
    HOME=/home/vllm \
    VLLM_NCCL_SO_PATH=/opt/vllm/lib/libnccl.so.2.18.1 \
    VLLM_USAGE_SOURCE=production-docker-image \
    VLLM_WORKER_MULTIPROC_METHOD=fork

# setup non-root user for OpenShift
RUN microdnf install -y shadow-utils \
    && umask 002 \
    && useradd --uid 2000 --gid 0 vllm \
    && microdnf remove -y shadow-utils \
    && microdnf clean all \
    && chmod g+rwx $HOME /usr/src /workspace

COPY LICENSE /licenses/vllm.md

USER 2000
CMD ["python", "-m", "vllm_tgis_adapter"]
