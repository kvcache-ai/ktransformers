#!/bin/bash
set -e  

# default backend
DEV="cuda"

# parse --dev argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dev) DEV="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
export DEV_BACKEND="$DEV"
echo "Selected backend: $DEV_BACKEND"

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

if [[ "$DEV_BACKEND" == "cuda" ]]; then
    echo "Installing custom_flashinfer for CUDA backend"
    pip install third_party/custom_flashinfer/
fi
# SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
# echo "Copying thirdparty libs to $SITE_PACKAGES"
# cp -a csrc/balance_serve/build/third_party/prometheus-cpp/lib/libprometheus-cpp-*.so* $SITE_PACKAGES/
# patchelf --set-rpath '$ORIGIN' $SITE_PACKAGES/sched_ext.cpython*

echo "Installation completed successfully"