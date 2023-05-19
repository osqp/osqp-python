#! /bin/bash

set -e
set -x

conda install -y -c intel mkl-devel
conda install -y -c conda-forge llvm-openmp
