@echo on

REM Needed to enable to define OSQP_DEPLOY_DIR within the file
@setlocal enabledelayedexpansion


REM Get OSQP version from local package
FOR /F "tokens=*" %%g IN ('python setup.py --version') do (SET OSQP_VERSION=%g)
IF NOT x%OSQP_VERSION%==x%OSQP_VERSION:dev=% (
set ANACONDA_LABEL="dev"
) ELSE (
set ANACONDA_LABEL="main"
)

REM Anaconda deploy
cd %APPVEYOR_BUILD_FOLDER%\conda_recipe

call conda build --python %PYTHON_VERSION% conda-recipe --output-folder conda-bld\
if errorlevel 1 exit /b 1

call anaconda -t %ANACONDA_TOKEN% upload conda-bld/**/osqp-*.tar.bz2 --user oxfordcontrol --force -l %ANACONDA_LABEL%

if errorlevel 1 exit /b 1

REM pypi deploy
cd %APPVEYOR_BUILD_FOLDER%
python setup.py bdist_wheel

IF "%TEST_PYPI%" == "true" (

twine upload --repository testpypi --config-file ci\pypirc -p %PYPI_PASSWORD% --skip-existing dist/*
if errorlevel 1 exit /b 1

) ELSE IF "%APPVEYOR_REPO_TAG%" == "true" (

twine upload --repository pypi --config-file ci\pypirc -p %PYPI_PASSWORD% dist/*
if errorlevel 1 exit /b 1

)

@echo off
