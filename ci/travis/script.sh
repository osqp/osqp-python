#!/bin/bash
set -e -x

cd ${TRAVIS_BUILD_DIR}

if [[ "${DISTRIB}" == "conda" ]]; then

    # Build and test with conda
    conda build conda-recipe --python=$PYTHON_VERSION --output-folder conda-bld

elif [[ "${DISTRIB}" == "pip" ]]; then
    


    if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
	echo "Creating pip binary package..."
	python setup.py bdist_wheel
    fi

    # Source distribution
    if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
	# Choose one python version to upload source distribution
	echo "Creating pip source package..."
	python setup.py sdist

        echo "Creating pip manylinux wheels package..."
        docker run --rm -e PLAT=$WHEELS_PLATFORM -v `pwd`:/io quay.io/pypa/$WHEELS_PLATFORM /io/ci/travis/build-wheels.sh
    fi

fi
