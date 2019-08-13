#!/bin/bash
set -e -x

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install cmake
    ln -s -f "${PYBIN}/cmake" /usr/bin/cmake
    cmake --version
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
