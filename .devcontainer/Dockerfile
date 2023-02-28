ARG CUDA=11.8
ARG RAPIDS=23.04
ARG DISTRO=ubuntu22.04
ARG REPO=rapidsai/devcontainers
ARG FROM_IMAGE=${REPO}:${RAPIDS}-cpp-cuda${CUDA}-mambaforge-${DISTRO}

FROM ${FROM_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive       \
 && sudo apt update                             \
 && sudo apt install -y --no-install-recommends \
    numactl rlwrap

USER coder

COPY --chown=coder:coder home/coder/workspace.code-workspace /home/coder/

COPY --chown=coder:coder opt/cunumeric /opt/cunumeric

WORKDIR /home/coder/

RUN /bin/bash -c 'mkdir -m 0755 -p ~/.{aws,cache,cargo,conda,config,local}'

ENV PYTHONDONTWRITEBYTECODE="1"

ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV SCCACHE_S3_KEY_PREFIX=legate-cunumeric-dev
ENV VAULT_HOST="https://vault.ops.k8s.rapids.ai"

ENV HISTFILE=/home/coder/.cache/._bash_history
