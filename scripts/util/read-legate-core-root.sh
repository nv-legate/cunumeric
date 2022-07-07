#! /usr/bin/env bash

# Read legate_core_ROOT from the environment or prompt the user to enter it
if [[ -z "$legate_core_ROOT" ]]; then
    while [[ -z "$legate_core_ROOT" || ! -d "$legate_core_ROOT" ]]; do
        read -ep "\`\$legate_core_ROOT\` not found.
Please enter the path to a legate.core build (or install) directory:
" legate_core_ROOT </dev/tty
    done
    echo "To skip this prompt next time, run:"
    echo "legate_core_ROOT=\"$legate_core_ROOT\" $1"
else
    echo "Using legate.core at: \`$legate_core_ROOT\`"
fi

export legate_core_ROOT="$(realpath -m "$legate_core_ROOT")"
