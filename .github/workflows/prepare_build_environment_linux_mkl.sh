#! /bin/bash

set -e
set -x

yum-config-manager --add-repo https://yum.repos.intel.com/oneapi
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
yum --nogpgcheck install -y intel-oneapi-mkl-devel-2023.0.0
