#! /usr/bin/env bash

mkdir -m 0755 -p ~/.{aws,cache,cargo,conda,config,local} ~/{legion,legate,cunumeric};

cat <<"EOF" >> ~/.bashrc
if [[ "$PATH" != *"/opt/cunumeric/bin"* ]]; then
    export PATH="$PATH:/opt/cunumeric/bin";
fi
EOF
