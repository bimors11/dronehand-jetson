#!/usr/bin/env bash

set -euo pipefail

sudo apt-get update -y

if ! command -v python3 &> /dev/null; then
    sudo apt-get install -y python3
else
    echo "Python 3 already installed: $(python3 --version)"
fi

if ! command -v pip3 &> /dev/null; then
    sudo apt-get install -y python3-pip
else
    echo "pip3 already installed: $(pip3 --version)"
fi

pip3 install --upgrade pip setuptools wheel

curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.11
source .venv/bin/activate

if [[ -f "requirements.txt" ]]; then
    uv pip install -r requirements.txt
else
    echo "No requirements.txt found in $(pwd)."
fi

echo "Python environment setup complete"
