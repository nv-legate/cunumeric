time ../legate.core/install/bin/legate -m cProfile -s tottime examples/testbench/test.py  --cpus 2 > temp_file && head -n 20 temp_file
