#!/bin/bash
set -e  

# clear build dirs
rm -rf ktransformers/ktransformers_ext/build
rm -rf ktransformers/ktransformers_ext/cuda/build
rm -rf ktransformers/ktransformers_ext/cuda/dist
rm -rf ktransformers/ktransformers_ext/cuda/*.egg-info

echo "Installing python dependencies from requirements.txt"
pip install -r requirements-local_chat.txt

echo "Installing ktransformers cpuinfer"
mkdir -p ktransformers/ktransformers_ext/build
cd ktransformers/ktransformers_ext/build
cmake ..
cmake --build . --config Release

echo "Installing ktransformers gpu kernel, this may take for a while, please wait"
sleep 3

cd ../cuda
python setup.py install
cd ../../..
echo "Installation completed successfully"