#! /bin/bash

set -e
set -x

curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda_11.6.1_511.65_windows.exe
./cuda.exe -s nvcc_11.6 cudart_11.6 cublas_dev_11.6 curand_dev_11.6 cusparse_dev_11.6 thrust_11.6 visual_studio_integration_11.6
rm cuda.exe
