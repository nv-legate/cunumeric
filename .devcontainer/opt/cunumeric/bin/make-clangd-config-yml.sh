#! /usr/bin/env bash

skbuild_dir="$(python -c 'import skbuild; print(skbuild.constants.SKBUILD_DIR())')";

mkdir -p ~/.config/clangd;

cat <<EOF >  ~/.config/clangd/config.yaml
$(cat /opt/cunumeric/.clangd)
---
If:
  PathMatch: $HOME/legate/.*
CompileFlags:
  CompilationDatabase: $HOME/legate/${skbuild_dir}/cmake-build
---
If:
  PathMatch: $HOME/cunumeric/.*
CompileFlags:
  CompilationDatabase: $HOME/cunumeric/${skbuild_dir}/cmake-build
EOF
