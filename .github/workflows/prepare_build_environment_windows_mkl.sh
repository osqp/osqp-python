#! /bin/bash

set -e
set -x

# See https://github.com/oneapi-src/oneapi-ci/blob/master/.github/workflows/build_all.yml for installer URLs
# WINDOWS_BASEKIT_URL is the one we want here
curl -L -nv -o webimage.exe https://registrationcenter-download.intel.com/akdlm/IRC_NAS/ae29263e-38b9-4d43-86c3-376d6e0668e7/intel-oneapi-base-toolkit-2025.0.1.47_offline.exe
./webimage.exe -s -x -f webimage_extracted --log extract.log
rm webimage.exe
./webimage_extracted/bootstrapper.exe -s --action install --components="intel.oneapi.win.mkl.devel" --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 --log-dir=.

pip install delvewheel

# -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64_win;C:/Program Files (x86)/oneDNN" -DBUILD_CLI=OFF -DWITH_DNNL=ON -DWITH_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2" -DCUDA_DYNAMIC_LOADING=ON -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all" -DCUDA_ARCH_LIST="Common" ..
# cp "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/redist/intel64_win/compiler/libiomp5md.dll" <deployment_dir>
