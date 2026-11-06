#!/bin/bash
set -euo pipefail
assignment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$assignment_root"
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Run this natively in an arm64 Terminal on your M1 Mac."
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  if [[ -x /opt/homebrew/bin/brew ]]; then
    export PATH="/opt/homebrew/bin:$PATH"
  else
    echo "Install Homebrew from https://brew.sh, then: brew install llvm libomp python"
    exit 1
  fi
fi
mac_python_prefix="$(brew --prefix python)"
mac_python="$mac_python_prefix/libexec/bin/python3"
if [[ ! -x "$mac_python" && -x "$mac_python_prefix/bin/python3" ]]; then
  mac_python="$mac_python_prefix/bin/python3"
fi
if [[ ! -x "$mac_python" ]]; then
  echo "Run: brew install llvm libomp python"
  exit 1
fi
if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i "$mac_python" src/macos_run.py "$@"
else
  exec "$mac_python" src/macos_run.py "$@"
fi
