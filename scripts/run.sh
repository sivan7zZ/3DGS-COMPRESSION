#!/usr/bin/env bash
# Run any local Python entry point using the existing unified environment.
set -euo pipefail
PWSR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PWSR_PYTHON="${PWSR_PYTHON:-/home/zhangys/anaconda3/envs/prune_wisely/bin/python}"
if [[ ! -x "$PWSR_PYTHON" ]]; then
  echo "Set PWSR_PYTHON to the Python executable in your installed environment." >&2
  exit 1
fi
PWSR_ENV_LIB="$(cd -- "$(dirname -- "$PWSR_PYTHON")/../lib" && pwd)"
PWSR_TORCH_LIB="$($PWSR_PYTHON -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
# This machine exposes libcuda.so.1 but not the unversioned dlopen name.
# Keep the alias inside this repository; never depend on historical result folders.
if [[ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 && ! -e /usr/lib/x86_64-linux-gnu/libcuda.so ]]; then
  mkdir -p "$PWSR_ROOT/.runtime/lib"
  if [[ ! -e "$PWSR_ROOT/.runtime/lib/libcuda.so" && ! -L "$PWSR_ROOT/.runtime/lib/libcuda.so" ]]; then
    ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$PWSR_ROOT/.runtime/lib/libcuda.so"
  fi
fi
export LD_LIBRARY_PATH="$PWSR_ROOT/.runtime/lib:$PWSR_ENV_LIB:$PWSR_TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd -- "$PWSR_ROOT"
if [[ $# -eq 0 ]]; then
  set -- run_pipeline.py
fi
exec "$PWSR_PYTHON" "$@"
