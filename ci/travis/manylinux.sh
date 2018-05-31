#!/bin/bash
set -e -x

# Package manylinux image

# Create wheels
for PYBIN in /opt/python/*/bin; do

	if [[ ${PYBIN} != *"34"* ]]; then

		# Install and link cmake
		${PYBIN}/pip install cmake
		rm -rf /usr/local/bin/cmake
		ln -s $PYBIN/cmake /usr/local/bin/cmake

		# Install python dependencies
		${PYBIN}/pip install numpy scipy future pytest

		# Create wheel
		${PYBIN}/pip wheel /io/ -w dist/

		# Bundle external shared libraries into the wheels
		for whl in dist/osqp*.whl; do
			auditwheel repair "$whl" -w /io/dist/
		done

		# Install package and test
		${PYBIN}/pip install osqp --no-index -f /io/dist

		# Test
		cd /io/
		${PYBIN}/pytest
	fi

done


