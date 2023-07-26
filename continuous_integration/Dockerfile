ARG BASE_IMAGE
FROM ${BASE_IMAGE} as stage0

COPY --chown=coder:coder continuous_integration/home/coder/.local/bin/* /home/coder/.local/bin/
COPY --chown=coder:coder . /home/coder/cunumeric

RUN chmod a+x /home/coder/.local/bin/*

#---------------------------------------------------
FROM stage0 as setup

USER coder
WORKDIR /home/coder

RUN get-yaml-and-make-conda-env

#---------------------------------------------------
FROM setup as build
USER coder
WORKDIR /home/coder

ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}
ARG AWS_SESSION_TOKEN
ENV AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

COPY --chown=coder:coder .creds /run/secrets

RUN entrypoint build-cunumeric-all

#---------------------------------------------------
FROM stage0 as final
USER coder
WORKDIR /home/coder

COPY --from=build --chown=coder:coder /tmp/out /tmp/out
COPY --from=build --chown=coder:coder /tmp/conda-build /tmp/conda-build
COPY --from=build --chown=coder:coder /tmp/env_yaml /tmp/env_yaml
