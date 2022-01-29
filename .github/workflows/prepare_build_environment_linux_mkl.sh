#! /bin/bash

set -e
set -x

pip install "cmake==3.22.*"

ONEAPI_VERSION=2022.0.1
MKL_BUILD=117
yum-config-manager --add-repo https://yum.repos.intel.com/oneapi
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
yum install -y intel-oneapi-mkl-devel-$ONEAPI_VERSION-$MKL_BUILD
echo "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin" > /etc/ld.so.conf.d/libiomp5.conf
ldconfig
