#!/bin/bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh

INITIALIZED="FALSE"
ROOT_DIR=$(pwd)

if [ -f ".INITIALIZED" ]; then
    INITIALIZED="TRUE"
fi

case "$INITIALIZED" in
    TRUE)
        echo "Detect file .INITIALIZED, will not init again."
        ;;
    FALSE)
        echo "Not detect file .INITIALIZED, do init."
        # Detect architecture
        ARCH=$(uname -m)
        IS_ARM=false
        if [[ "$ARCH" == "armv7l" || "$ARCH" == "aarch64" ]]; then
            IS_ARM=true
        fi

        # ARM-specific operations
        if $IS_ARM; then
            echo "Processing ARM architecture specific tasks"

            # Copy ARM specific files
            # cp ./for_arm/CMakeLists.txt ./csrc/ktransformers_ext/CMakeLists.txt
            # cp ./for_arm/iqk_mul_mat.inc ./third_party/llamafile/iqk_mul_mat.inc
            # cp ./for_arm/sgemm.cpp ./third_party/llamafile/sgemm.cpp
            # cp ./for_arm/tinyblas_cpu_sgemm.inc ./third_party/llamafile/tinyblas_cpu_sgemm.inc
            cp ./for_arm/requirements-local_chat.txt ./requirements-local_chat.txt
            cp ./for_arm/setup.py ./setup.py
        fi

        # init third_party
        # clone third_party or unzip third_party file
        third_party_file=""
        third_party="$ROOT_DIR/third_party"
        cd "$third_party"

        for i in "$@"
        do
        case $i in
            --third-party-file=*|-f=*)
            third_party_file="${i#*=}"
            shift
            ;;
            *)
            echo "Unknown operation: $i"
            exit 1
            ;;
        esac
        done

        if [ -n "$third_party_file" ]; then
            if [[ "$third_party_file" != /* ]]; then
                third_party_file="$ROOT_DIR/$third_party_file"
            fi

            if [ ! -f "$third_party_file" ]; then
                echo "Error: file not found on '$third_party_file'"
                exit 1
            fi

            case "${third_party_file}" in
                *.tar.gz|*.tgz)
                    tar -xzf "$third_party_file"
                    ;;
                *.zip)
                    unzip "$third_party_file"
                    ;;
                *)
                    echo "Error: unsupported file format '$third_party_file'"
                    exit 1
                    ;;
            esac
            echo "Finish decompress ${third_party_file}"
        else
            # todo update
            git clone https://github.com/kvcache-ai/custom_flashinfer.git -b fix-precision-mla-merge-main && cd custom_flashinfer && git checkout fd94393f
            git submodule init && git submodule update && cd 3rdparty
            cd composable_kernels && git checkout 5055b3bd && cd ..
            cd cutlass && git checkout cc3c29a8 && cd ..
            cd googletest && git checkout 5a37b517 && cd ..
            cd mscclpp && git checkout v0.5.1 && cd ..
            cd nvbench && git checkout 555d628e && cd ..
            cd spdlog && git checkout v1.x && cd ..
            cd "$third_party"

            git clone https://github.com/ggerganov/llama.cpp.git -b master && cd llama.cpp && git checkout b3173
            git submodule init && git submodule update
            cd kompute && git checkout 4565194e && cd ..
            cd "$third_party"

            git clone https://github.com/jupp0r/prometheus-cpp -b master && cd prometheus-cpp && git checkout f13cdd05
            git submodule init && git submodule update && cd 3rdparty
            cd civetweb && git checkout v1.16 && cd ..
            cd googletest && git checkout release-1.11.0 && cd ..
            cd "$third_party"

            git clone https://github.com/pybind/pybind11.git -b master && cd pybind11 && git checkout bb05e081 && cd ..
            git clone https://github.com/gabime/spdlog.git -b v1.x && cd spdlog && git checkout v1.15.2 && cd ..
            git clone https://github.com/Cyan4973/xxHash.git -b dev && cd xxHash && git checkout 953a09ab && cd ..

            echo "Finish clone and checkout third_party"
        fi

        cd "$ROOT_DIR"
        touch ./.INITIALIZED
        ;;
    *)
        echo "Error"
        exit 1
        ;;
esac

cd "$ROOT_DIR"
sed -i 's/\r$//' ./install.sh
bash ./install.sh
