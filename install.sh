#!/usr/bin/env bash

set -euo pipefail

sudo apt-get update -y

sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

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

uv venv --python 3.10
source .venv/bin/activate

if [[ -f "requirements.txt" ]]; then
    uv pip install -r requirements.txt
else
    echo "No requirements.txt found in $(pwd)."
fi

echo "Environment setup complete"
