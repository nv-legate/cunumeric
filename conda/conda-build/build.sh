# We rely on an environment variable to determine if we need to build cpu-only bits
if [ -z "$CPU_ONLY" ]; then
  cp $PREFIX/lib/stubs/libcuda.so $PREFIX/lib/libcuda.so
  ln -s $PREFIX/lib $PREFIX/lib64
  $PYTHON install.py --with-core $PREFIX --with-cutensor $PREFIX -v
  rm $PREFIX/lib/libcuda.so
  rm $PREFIX/lib64
else
  $PYTHON install.py --with-core $PREFIX --with-cutensor $PREFIX -v
fi
