#! /usr/bin/env bash

rm -rf $(find "$CONDA_PREFIX/lib" -type d -name '*cunumeric*') \
       $(find "$CONDA_PREFIX/lib" -type f -name 'libcunumeric*') \
       $(find "$CONDA_PREFIX/include" -type d -name '*cunumeric*') \
       ;
