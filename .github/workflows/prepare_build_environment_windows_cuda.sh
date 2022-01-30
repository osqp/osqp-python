#! /bin/bash

set -e
set -x

curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_461.33_win10.exe
#curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.89_win10.exe
#curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda_11.4.1_471.41_win10.exe
#curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_511.23_windows.exe

./cuda.exe -s nvcc_11.2 cudart_11.2 cublas_dev_11.2 curand_dev_11.2 cusparse_dev_11.2 visual_studio_integration_11.2
#./cuda.exe -s nvcc_11.3 cudart_11.3 cublas_dev_11.3 curand_dev_11.3 cusparse_dev_11.3 visual_studio_integration_11.3
#./cuda.exe -s nvcc_11.4 cudart_11.4 cublas_dev_11.4 curand_dev_11.4 cusparse_dev_11.4 visual_studio_integration_11.4
#./cuda.exe -s nvcc_11.6 cudart_11.6 cublas_dev_11.6 curand_dev_11.6 cusparse_dev_11.6 visual_studio_integration_11.6
rm cuda.exe
