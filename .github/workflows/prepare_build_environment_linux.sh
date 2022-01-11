#! /bin/bash

set -e
set -x

pip install "cmake==3.22.*"

if [ "$CIBW_ARCHS" == "aarch64" ]; then

    OPENBLAS_VERSION=0.3.19
    curl -L -O https://github.com/xianyi/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz
    tar xf *.tar.gz && rm *.tar.gz
    cd OpenBLAS-*
    # NUM_THREADS: maximum value for intra_threads
    # NUM_PARALLEL: maximum value for inter_threads
    make TARGET=ARMV8 NO_SHARED=1 BUILD_SINGLE=1 NO_LAPACK=1 ONLY_CBLAS=1 USE_OPENMP=1 NUM_THREADS=32 NUM_PARALLEL=8
    make install NO_SHARED=1
    cd ..
    rm -r OpenBLAS-*

else

    # Install CUDA 11.2, see:
    # * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/base/Dockerfile
    # * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/devel/Dockerfile
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
    yum install --setopt=obsoletes=0 -y \
        cuda-nvcc-11-2-11.2.152-1 \
        cuda-cudart-devel-11-2-11.2.152-1 \
        libcurand-devel-11-2-10.2.3.152-1 \
        libcublas-devel-11-2-11.4.1.1043-1
    ln -s cuda-11.2 /usr/local/cuda

    ONEAPI_VERSION=2022.0.1
    MKL_BUILD=117
    yum-config-manager --add-repo https://yum.repos.intel.com/oneapi
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    yum install -y intel-oneapi-mkl-devel-$ONEAPI_VERSION-$MKL_BUILD
    echo "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin" > /etc/ld.so.conf.d/libiomp5.conf
    ldconfig

fi

mkdir build-release && cd build-release

if [ "$CIBW_ARCHS" == "aarch64" ]; then
else
fi

cd ..
rm -r build-release
