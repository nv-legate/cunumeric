cp $PREFIX/lib/stubs/libcuda.so $PREFIX/lib/libcuda.so
ln -s $PREFIX/lib $PREFIX/lib64
$PYTHON install.py --with-core $PREFIX --with-cutensor $PREFIX -v
rm $PREFIX/lib/libcuda.so
rm $PREFIX/lib64
