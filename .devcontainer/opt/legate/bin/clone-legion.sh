#! /usr/bin/env bash

if [[ ! -d ~/legion/.git ]]; then
    echo "Cloning Legion" 1>&2;
    /opt/devcontainer/bin/gitlab/repo/clone.sh "StanfordLegion" "legion";
fi
