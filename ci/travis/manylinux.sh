#!/bin/bash
set -e -x

# Package manylinux image

# Create wheels
for PYBIN in /opt/python/*/bin; do

	# Install and link cmake
	${PYBIN}/pip install cmake
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
	(cd /io/; ${PYBIN}/pytest -x)

done


