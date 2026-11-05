#!/usr/bin/env bash
# Run from any directory: bash environment/setup_linux.sh [new-venv-path]
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != Linux ]]; then
  echo 'Use Linux or a Linux terminal inside WSL2.' >&2
  exit 1
fi
venv_path="${1:-.venv-mango}"
if [[ -e "$venv_path" ]]; then
  echo "Refusing to replace an existing path: $venv_path" >&2
  exit 1
fi
python3.12 -m venv "$venv_path"
py="$venv_path/bin/python"
"$py" -m pip install 'pip==26.1.2' 'setuptools==78.1.0'
"$py" -m pip install \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -c environment/asus/requirements-host-freeze.txt \
  'torch==2.13.0+cu130' 'torchvision==0.28.0+cu130' \
  -r environment/requirements-training.txt

"$py" -m pip install 'wheel==0.47.0'
"$py" environment/fix_cusparselt_wheel.py

if [[ "$(uname -m)" == aarch64 && -f environment/asus/wheels/mamba_ssm-2.3.2.post1-cp312-cp312-linux_aarch64.whl ]]; then
  "$py" -m pip install --no-deps \
    environment/asus/wheels/mamba_ssm-2.3.2.post1-cp312-cp312-linux_aarch64.whl
else
  : "${CUDA_HOME:=/usr/local/cuda-13.0}"
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  nvcc --version
  MAMBA_FORCE_BUILD=TRUE MAX_JOBS=2 "$py" -m pip install \
    --no-build-isolation --no-deps 'mamba-ssm==2.3.2.post1'
fi
MPLBACKEND=Agg "$py" code/check_environment.py
"$py" -m pip check
