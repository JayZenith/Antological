#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x /venv/main/bin/python ]]; then
  python_bin=/venv/main/bin/python
else
  python_bin="$(command -v python3)"
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install \
    --python "$python_bin" \
    --only-binary=:all: \
    -r requirements.txt
else
  "$python_bin" -m pip install \
    --only-binary=:all: \
    -r requirements.txt
fi

mkdir -p direct_session/attempts analogy_session/attempts
PYTHONDONTWRITEBYTECODE=1 "$python_bin" validate.py

"$python_bin" - <<'PY'
import torch
import torchvision

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
