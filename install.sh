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
pip install -r requirements-local_chat.txt

echo "Installing ktransformers"
KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation
echo "Installation completed successfully"