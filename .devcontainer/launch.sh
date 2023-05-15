#! /usr/bin/env bash

launch_devcontainer() {

    # Ensure we're in the repo root
    cd "$( cd "$( dirname "$(realpath -m "${BASH_SOURCE[0]}")" )" && pwd )/..";

    local mode="${1:-unified}";

    case "$mode" in
        unified ) ;;
        isolated) ;;
        *      ) mode="unified";;
    esac

    local flavor="conda/${mode}";
    local workspace="$(basename "$(pwd)")";
    local tmpdir="$(mktemp -d)/${workspace}";
    local path="$(pwd)/.devcontainer/${flavor}";

    mkdir -p "${tmpdir}";
    cp -arL "$path/.devcontainer" "${tmpdir}/";
    sed -i "s@\${localWorkspaceFolder}@$(pwd)@g" "${tmpdir}/.devcontainer/devcontainer.json";
    path="${tmpdir}";

    local hash="$(echo -n "${path}" | xxd -pu - | tr -d '[:space:]')";
    local url="vscode://vscode-remote/dev-container+${hash}/home/coder";

    echo "devcontainer URL: ${url}";

    local launch="";
    if type open >/dev/null 2>&1; then
        launch="open";
    elif type xdg-open >/dev/null 2>&1; then
        launch="xdg-open";
    fi

    if [ -n "${launch}" ]; then
        code --new-window "${tmpdir}";
        $launch "${url}" >/dev/null 2>&1 &
    fi
}

launch_devcontainer "$@";
