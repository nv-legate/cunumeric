#! /usr/bin/env bash

rm -rf $(find "$CONDA_PREFIX/lib" -mindepth 1 -type d -name '*cunumeric*') \
       $(find "$CONDA_PREFIX/lib" -mindepth 1 -type f -name 'libcunumeric*') \
       $(find "$CONDA_PREFIX/lib" -mindepth 1 -type f -name 'cunumeric.egg-link') \
       $(find "$CONDA_PREFIX/include" -mindepth 1 -type f -name 'tci.h') \
       $(find "$CONDA_PREFIX/include" -mindepth 1 -type d -name 'tci') \
       $(find "$CONDA_PREFIX/include" -mindepth 1 -type d -name 'tblis') \
       $(find "$CONDA_PREFIX/include" -mindepth 1 -type d -name 'cunumeric') \
       ;
