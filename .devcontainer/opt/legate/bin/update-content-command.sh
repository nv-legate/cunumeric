#! /usr/bin/env bash

mkdir -m 0755 -p ~/{.aws,.cache,.cargo,.conda,.config,legion,legate,cunumeric};

cat <<"EOF" >> ~/.bashrc
if [[ "$PATH" != *"/opt/legate/bin"* ]]; then
    export PATH="$PATH:/opt/legate/bin";
fi
EOF
