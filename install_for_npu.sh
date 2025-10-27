#!/bin/bash
set -e  
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# clear build dirs
rm -rf build
rm -rf *.egg-info
rm -rf csrc/build
rm -rf csrc/ktransformers_ext/build
rm -rf csrc/ktransformers_ext/cuda/build
rm -rf csrc/ktransformers_ext/cuda/dist
rm -rf csrc/ktransformers_ext/cuda/*.egg-info
rm -rf ~/.ktransformers
echo "Installing python dependencies from requirements.txt"
pip install -r requirements-local_chat.txt
pip install -r ktransformers/server/requirements.txt
echo "Installing ktransformers"
KTRANSFORMERS_FORCE_BUILD=TRUE pip install -v . --no-build-isolation
pip install third_party/custom_flashinfer/

echo "Installation completed successfully"
