#!/bin/bash

cd "${0%/*}"
git submodule update --init --recursive

sudo apt update
sudo apt install libtbb-dev
sudo apt install libcurl4-openssl-dev
sudo apt install libaio-dev

cd third_party/xxHash/
make -j
sudo make install
cd ../..

