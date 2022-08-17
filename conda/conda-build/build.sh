install_args=()

# We rely on an environment variable to determine if we need to build cpu-only bits
if [ -z "$CPU_ONLY" ]; then
  # cutensor, relying on the conda cutensor package
  install_args+=("--with-cutensor" "$PREFIX")
else
  # When we build without cuda, we need to provide the location of curand
  install_args+=("--with-curand" "$PREFIX")
fi

# location of legate-core
install_args+=("--with-core" "$PREFIX")

# location of openblas, relying on the conda openblas package
install_args+=("--with-openblas" "$PREFIX")

# Verbose mode
install_args+=("-v")

# Move the stub library into the lib package to make the install think it's pointing at a live installation
if [ -z "$CPU_ONLY" ]; then
  cp $PREFIX/lib/stubs/libcuda.so $PREFIX/lib/libcuda.so
  ln -s $PREFIX/lib $PREFIX/lib64
fi

echo "Install command: $PYTHON install.py ${install_args[@]}"
$PYTHON install.py "${install_args[@]}"

# Remove the stub library and linking
if [ -z "$CPU_ONLY" ]; then
  rm $PREFIX/lib/libcuda.so
  rm $PREFIX/lib64
fi
