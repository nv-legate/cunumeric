#!/bin/bash
#touch src/matrix/matvecmul.cc
#touch src/fused/fused_op.cc
cd ../legate.core/
python setup.py --no-cuda
cd ../legate.numpy/
python setup.py --with-core /Users/shivsundram/Desktop/coding/legate/legate.core/install/ --with-openblas /usr/local/opt/openblas/ -j 4 --verbose
#../legate.core/install/bin/legate examples/testbench/test.py --cpus 2
