#! /usr/bin/env bash

generate_conda_envs_py="legate/scripts/generate-conda-envs.py";

if [[ ! -d ~/legate/.git ]]; then
    echo "Cloning legate.core" 1>&2;
    /opt/devcontainer/bin/github/repo/clone.sh "nv-legate" "legate.core" "legate";
fi

if [[ ! -f ~/$generate_conda_envs_py ]]; then
    default_branch="$(\
        gh repo list nv-legate \
            --json name --json defaultBranchRef \
            --jq '. | map(select(.name == "legate.core")) | map(.defaultBranchRef.name)[]' \
        )";
    echo ">> ~/$generate_conda_envs_py not found" 1>&2;
    echo ">> switching to branch '$default_branch'" 1>&2;
    (
        cd ~/legate && git fetch upstream && git checkout "$default_branch";
    )
fi
