#! /usr/bin/env bash

if [[ ! -d ~/legion/.git ]]; then
    echo "Cloning Legion" 1>&2;
    gitlab-repo-clone "StanfordLegion" "legion";
fi
