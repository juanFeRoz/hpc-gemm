FROM rocm/dev-ubuntu-22.04:5.7

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget gnupg software-properties-common \
    git cmake ninja-build build-essential \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 17

RUN apt-get update && apt-get install -y \
    libclang-17-dev \
    clang-tools-17 \
    libomp-17-dev \
    llvm-17-dev \
    lld-17 \
    && rm -rf /var/lib/apt/lists/*

ENV LLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm
ENV PATH=/usr/lib/llvm-17/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib/llvm-17/lib:$LD_LIBRARY_PATH

ENV ROCM_PATH=/opt/rocm

WORKDIR /opt
RUN git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git

WORKDIR /opt/AdaptiveCpp/build

RUN cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_ROCM_BACKEND=ON \
  -DLLVM_DIR=$LLVM_DIR \
  -DCMAKE_C_COMPILER=clang-17 \
  -DCMAKE_CXX_COMPILER=clang++-17 \
  -DCMAKE_INSTALL_PREFIX=/opt/acpp-install

RUN ninja -j$(nproc)
RUN ninja install

ENV PATH=/opt/acpp-install/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/acpp-install/lib:$LD_LIBRARY_PATH

WORKDIR /workspace

CMD ["/bin/bash"]