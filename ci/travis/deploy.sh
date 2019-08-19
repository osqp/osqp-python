#!/bin/bash
set -e -x

# Deploy packages

# Get OSQP version
export OSQP_VERSION=`python setup.py --version`
if [[ ${OSQP_VERSION} == *"dev"* ]]; then
    export TEST_PYPI="true"
    export ANACONDA_LABEL="dev";
else
    export TEST_PYPI="false"
    export ANACONDA_LABEL="main";
fi

if [[ "${DISTRIB}" == "conda" ]]; then
    
# Anaconda
echo "Deploying to Anaconda..."
anaconda -t $ANACONDA_TOKEN upload ${TRAVIS_BUILD_DIR}/conda-bld/**/*.tar.bz2 --skip-existing --user oxfordcontrol -l ${ANACONDA_LABEL}

fi

if [[ "${DISTRIB}" == "pip" ]]; then
    if [[ -d "dist" && -n "$(ls -A dist)" ]]; then
	echo "Deploying to Pypi..."
	if [[ "$TEST_PYPI" == "true" ]]; then
	    twine upload --repository testpypi --config-file ci/pypirc -p $PYPI_PASSWORD --skip-existing dist/osqp-*     # Test pypi repo
	else
	    # Upload to main pypi repo if it is not dev and it is a tag
	    twine upload --repository pypi --config-file ci/pypirc -p $PYPI_PASSWORD dist/osqp-*
	fi
	echo "Successfully deployed to Pypi"
fi

fi
