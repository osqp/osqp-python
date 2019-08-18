#!/bin/bash
set -e -x

# Remove version 34 (not supported)
rm -rf /opt/python/cp34*

# Install cmake
# Use version 35 for cmake. The pip version
# of cmake is not compatible if we build the wheels
# with the same version. Use CMAKE_PIP_BIN_ALT for other version
CMAKE_PIP_BIN=/opt/python/cp35-cp35m/bin
CMAKE_PIP_BIN_ALT=/opt/python/cp36-cp36m/bin
"${CMAKE_PIP_BIN}/pip" install cmake
"${CMAKE_PIP_BIN_ALT}/pip" install cmake
ln -snf "${CMAKE_PIP_BIN}/cmake" /usr/bin/cmake
cmake --version


# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ $PYBIN == *"35"* ]]; then
    	# Fix with cmake and same python version
        ln -snf "${CMAKE_PIP_BIN_ALT}/cmake" /usr/bin/cmake
        cmake --version
    fi

    "${PYBIN}/pip" install --upgrade pip
    "${PYBIN}/pip" install -r /io/requirements.txt
    "${PYBIN}/pip" install pytest
    "${PYBIN}/pip" wheel /io/ -w dist/
    
    if [[ ${PYBIN} == *"35"* ]]; then
    	# Fix symbolic link back
        ln -snf "${CMAKE_PIP_BIN}/cmake" /usr/bin/cmake
    fi

done

for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/dist/
done

for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install osqp --no-index -f /io/dist
    # Disable MKL tests since MKL is not in the docker image
    cd "$HOME"
    "${PYBIN}/python" -m pytest --pyargs osqp -k 'not mkl_'
done
