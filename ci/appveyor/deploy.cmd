@echo on

REM Needed to enable to define OSQP_DEPLOY_DIR within the file
@setlocal enabledelayedexpansion

IF "%APPVEYOR_REPO_TAG%" == "true" (

REM Anaconda deploy
cd %APPVEYOR_BUILD_FOLDER%\conda_recipe

call conda build --python %PYTHON_VERSION% osqp --output-folder conda-bld\
if errorlevel 1 exit /b 1

call anaconda -t %ANACONDA_TOKEN% upload conda-bld/**/osqp-*.tar.bz2 --user oxfordcontrol
if errorlevel 1 exit /b 1

REM pypi deploy
cd %APPVEYOR_BUILD_FOLDER%
call activate test-environment
python setup.py bdist_wheel
IF "%TEST_PYPI%" == "true" (
    twine upload --repository testpypi --config-file ci\pypirc -p %PYPI_PASSWORD% dist/*
    if errorlevel 1 exit /b 1
) ELSE (
    twine upload --repository pypi --config-file ci\pypirc -p %PYPI_PASSWORD% dist/*
    if errorlevel 1 exit /b 1
)

REM Close parenthesis for deploying only if it is a tagged commit
)
@echo off

