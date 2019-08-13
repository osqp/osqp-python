#!/bin/bash
set -e -x

# Install cmake
CMAKE_PIP_BIN=/opt/python/cp37-cp37m/bin
# "${CMAKE_PIP_BIN}/pip" install --upgrade pip
"${CMAKE_PIP_BIN}/pip" install cmake
ln -s "${CMAKE_PIP_BIN}/cmake" /usr/bin/cmake
cmake --version

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    # "${PYBIN}/pip" install --upgrade pip
    "${PYBIN}/pip" install pytest
    "${PYBIN}/pip" wheel /io/ -w dist/
done

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/dist/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install osqp --no-index -f /io/dist/
    (cd "$HOME"; "${PYBIN}/python" -m pytest osqp)
done
