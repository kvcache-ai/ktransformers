#!/bin/bash
set -e

# clear build dirs
rm -rf build
rm -rf *.egg-info
rm -rf ktransformers/ktransformers_ext/build
rm -rf ktransformers/ktransformers_ext/cuda/build
rm -rf ktransformers/ktransformers_ext/cuda/dist
rm -rf ktransformers/ktransformers_ext/cuda/*.egg-info

echo "Installing python dependencies from requirements.txt"
if command -v mcc > /dev/null 2>&1; then
    bash -c 'pip install -r <(grep -v -E "torch|numpy" requirements-local_chat.txt)'
else
    pip install -r requirements-local_chat.txt
fi

echo "Installing ktransformers"
KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation
echo "Installation completed successfully"