@echo on

cd %APPVEYOR_BUILD_FOLDER%

REM Get OSQP version from local package
FOR /F "tokens=*" %%g IN ('python setup.py --version') do (SET OSQP_VERSION=%%g)
IF NOT x%OSQP_VERSION%==x%OSQP_VERSION:dev=% (
set ANACONDA_LABEL="dev"
set TEST_PYPI="true"
) ELSE (
set ANACONDA_LABEL="main"
set TEST_PIPI="false"
)
ECHO %ANACONDA_LABEL%

REM Needed to enable to define OSQP_DEPLOY_DIR within the file

@setlocal enabledelayedexpansion

IF "%DISTRIB%"=="conda" (

REM Anaconda deploy
call anaconda -t %ANACONDA_TOKEN% upload conda-bld/**/osqp-*.tar.bz2 --user oxfordcontrol --force -l %ANACONDA_LABEL%
if errorlevel 1 exit /b 1

) ELSE IF "%DISTRIB%"=="pip" (

REM pypi deploy

IF %TEST_PYPI% == "true" (

twine upload --repository testpypi --config-file ci\pypirc -p %PYPI_PASSWORD% --skip-existing dist/*
if errorlevel 1 exit /b 1

) ELSE IF %APPVEYOR_REPO_TAG% == "true" (

twine upload --repository pypi --config-file ci\pypirc -p %PYPI_PASSWORD% dist/*
if errorlevel 1 exit /b 1

)

)
@echo off
