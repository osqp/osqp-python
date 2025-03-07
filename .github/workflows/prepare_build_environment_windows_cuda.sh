#! /bin/bash

set -e
set -x

curl -s -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe
./cuda.exe -s nvcc_12.5 cudart_12.5 cublas_dev_12.5 curand_dev_12.5 cusparse_dev_12.5 thrust_12.5 visual_studio_integration_12.5
rm cuda.exe
