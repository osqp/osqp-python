#! /bin/bash

set -e
set -x

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install -y cuda-toolkit-12-4

/usr/local/cuda-12.4/bin/nvcc --version
