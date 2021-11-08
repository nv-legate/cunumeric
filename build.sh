#!/bin/bash
#touch src/matrix/matvecmul.cc
#touch src/fused/fused_op.cc
cd ../legate.core/
python setup.py --no-cuda 
cd ../legate.numpy/
python setup.py --clean --with-core ~/legate/legate.core/install/ --verbose
#../legate.core/install/bin/legate examples/testbench/test.py --cpus 2
