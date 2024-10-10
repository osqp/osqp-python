#! /bin/bash

set -e
set -x

# Install OneAPI MKL
# See https://github.com/oneapi-src/oneapi-ci for installer URLs
ONEAPI_INSTALLER_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/cd013e6c-49c4-488b-8b86-25df6693a9b7/m_BaseKit_p_2023.2.0.49398.dmg
wget -q $ONEAPI_INSTALLER_URL
hdiutil attach -noverify -noautofsck $(basename $ONEAPI_INSTALLER_URL)
sudo /Volumes/$(basename $ONEAPI_INSTALLER_URL .dmg)/bootstrapper.app/Contents/MacOS/bootstrapper --silent --eula accept --components intel.oneapi.mac.mkl.devel

# Install LLVM's libomp because Intel's OpenMP runtime included in MKL does
# not ship with the header file.
# brew install libomp

pip install "cmake==3.18.4"
