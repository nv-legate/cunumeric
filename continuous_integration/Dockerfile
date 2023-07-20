ARG BASE_IMAGE
FROM ${BASE_IMAGE} as stage0

COPY --chown=coder:coder continuous_integration/home/coder/.local/bin/* /home/coder/.local/bin/
COPY --chown=coder:coder . /home/coder/cnumeric

RUN chmod a+x /home/coder/.local/bin/*

#---------------------------------------------------
FROM stage0 as setup

RUN get-yaml-and-make-conda-env

#---------------------------------------------------
FROM setup as build
ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}
ARG AWS_SESSION_TOKEN
ENV AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

# If .creds exists copy it to /run/secrets
COPY --chown=coder:coder .cred[s] /run/secrets

RUN entrypoint build-cunumeric-all

#---------------------------------------------------
FROM stage0 as final
COPY --from=build /tmp/out /tmp/out
COPY --from=build /tmp/conda-build /tmp/conda-build
COPY --from=build /tmp/env_yaml /tmp/env_yaml
