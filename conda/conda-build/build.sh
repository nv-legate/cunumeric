install_args=()

# We rely on an environment variable to determine if we need to build cpu-only bits
if [ -z "$CPU_ONLY" ]; then
  # cutensor, relying on the conda cutensor package
  install_args+=("--with-cutensor" "$PREFIX")
fi

# location of legate-core
install_args+=("--with-core" "$PREFIX")

# location of openblas, relying on the conda openblas package
install_args+=("--with-openblas" "$PREFIX")

# Verbose mode
install_args+=("-v")

echo "Install command: $PYTHON install.py ${install_args[@]}"
$PYTHON install.py "${install_args[@]}"
