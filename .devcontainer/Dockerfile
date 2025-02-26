FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel as compile_server
WORKDIR /workspace
ENV CUDA_HOME /usr/local/cuda
RUN <<EOF
apt update -y &&  apt install -y  --no-install-recommends \
    git \
    wget \
    vim \
    gcc \
    g++ \
    cmake && 
rm -rf /var/lib/apt/lists/* &&
pip install ninja pyproject numpy cpufeature &&
pip install flash-attn &&
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/
EOF
# Set the default shell to bash
CMD ["/bin/bash"]