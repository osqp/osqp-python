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

REM  Removed pypi deployment (replaced by repository oxfordcontrol/osqp-wheels)
REM  REM pypi deploy
REM  cd %APPVEYOR_BUILD_FOLDER%
REM  call activate test-environment
REM  python setup.py bdist_wheel
REM  IF "%TEST_PYPI%" == "true" (
REM      twine upload --repository testpypi --config-file ci\pypirc -p %PYPI_PASSWORD% dist/*
REM      if errorlevel 1 exit /b 1
REM  ) ELSE (
REM      twine upload --repository pypi --config-file ci\pypirc -p %PYPI_PASSWORD% dist/*
REM      if errorlevel 1 exit /b 1
REM  )

REM Close parenthesis for deploying only if it is a tagged commit
)
@echo off

