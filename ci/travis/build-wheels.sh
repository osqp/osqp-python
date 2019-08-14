#!/bin/bash
set -e -x

# Install cmake
# Use version 35 for cmake. The pip version
# of cmake is not compatible if we build the wheels
# with the same version. Use CMAKE_PIP_BIN_ALT for other version
CMAKE_PIP_BIN=/opt/python/cp35-cp35m/bin
CMAKE_PIP_BIN_ALT=/opt/python/cp36-cp36m/bin
"${CMAKE_PIP_BIN}/pip" install cmake
"${CMAKE_PIP_BIN_ALT}/pip" install cmake
ln -s "${CMAKE_PIP_BIN}/cmake" /usr/bin/cmake
cmake --version

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ $PYBIN == *"35"* ]]; then
	# Fix with cmake and same python version
        ln -f -s "${CMAKE_PIP_BIN_ALT}/cmake" /usr/bin/cmake
        cmake --version
    fi
    
    "${PYBIN}/pip" install pytest
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/

    
    if [[ $PYBIN == *"35"* ]]; then
	# Fix with cmake and same python version
        ln -f -s "${CMAKE_PIP_BIN}/cmake" /usr/bin/cmake
        
    fi


done


# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install osqp --no-index -f /io/wheelhouse/
    (cd "$HOME"; "${PYBIN}/python" -m pytest osqp)
done
