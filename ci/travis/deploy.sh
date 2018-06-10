#!/bin/bash
set -ev


# Update variables from install
# CMake
if [[ "${TRAVIS_OS_NAME}" == "linux" ]]; then
    export PATH=${DEPS_DIR}/cmake/bin:${PATH}
fi
# Anaconda
export PATH=${DEPS_DIR}/miniconda/bin:$PATH
hash -r
source activate testenv



# Anaconda
cd ${TRAVIS_BUILD_DIR}/conda_recipe

echo "Creating conda package..."
conda build --python ${PYTHON_VERSION} osqp --output-folder conda-bld/

echo "Deploying to Anaconda.org..."
anaconda -t $ANACONDA_TOKEN upload conda-bld/**/osqp-*.tar.bz2 --user oxfordcontrol

echo "Successfully deployed to Anaconda.org."


# Removed pypi deployment (replaced by repository oxfordcontrol/osqp-wheels)
# # NB: Binary Linux packages not supported on Pypi
# cd ${TRAVIS_BUILD_DIR}
#
# if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
#
#         echo "Creating pip binary package..."
#         python setup.py bdist_wheel
#
#
#         echo "Deploying to Pypi..."
#         if [[ "$TEST_PYPI" == "true" ]]; then
#             twine upload --repository testpypi --config-file ci/pypirc -p $PYPI_PASSWORD dist/*     # Test pypi repo
#         else
#             twine upload --repository pypi --config-file ci/pypirc -p $PYPI_PASSWORD dist/*         # Main pypi repo
#         fi
#         echo "Successfully deployed to Pypi"
#
# else if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
#
#         # Create manylinux wheel
#         echo "Creating pip manylinux package..."
#
#         # Install docker
#         docker pull $MANYLINUX_DOCKER_IMAGE
#
#         # Run docker image
#         docker run --rm -v `pwd`:/io $MANYLINUX_DOCKER_IMAGE /io/ci/travis/manylinux.sh
#
#         echo "Deploying to Pypi..."
#         if [[ "$TEST_PYPI" == "true" ]]; then
#                 twine upload --repository testpypi --config-file ci/pypirc -p $PYPI_PASSWORD dist/*     # Test pypi repo
#         else
#                 twine upload --repository pypi --config-file ci/pypirc -p $PYPI_PASSWORD dist/*         # Main pypi repo
#         fi
#         echo "Successfully deployed to Pypi"
#
# fi

exit 0
