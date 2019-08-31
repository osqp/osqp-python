@echo on

@setlocal enabledelayedexpansion

REM  IF "%APPVEYOR_REPO_TAG%" == "true" (

cd %APPVEYOR_BUILD_FOLDER%

REM Get OSQP version from local package
FOR /F "tokens=*" %%g IN ('python setup.py --version') do (SET OSQP_VERSION=%%g)
IF NOT x!OSQP_VERSION!==x!OSQP_VERSION:dev=! (
set ANACONDA_LABEL=dev
set TEST_PYPI=true
) ELSE (
set ANACONDA_LABEL=main
set TEST_PIPI=false
)
ECHO %ANACONDA_LABEL%

IF "%DISTRIB%"=="conda" (
call anaconda -t %ANACONDA_TOKEN% upload conda-bld/**/osqp-*.tar.bz2 --skip-existing --user oxfordcontrol -l !ANACONDA_LABEL!
if errorlevel 1 exit /b 1
)


IF "%DISTRIB%"=="pip" (
IF "!TEST_PYPI!" == "true" (
twine upload --repository testpypi --config-file ci\pypirc -p %PYPI_PASSWORD% --skip-existing dist/osqp-*
if errorlevel 1 exit /b 1
) ELSE (
twine upload --repository pypi --config-file ci\pypirc -p %PYPI_PASSWORD% --skip-existing dist/osqp-*
if errorlevel 1 exit /b 1
)
)


REM  rem Deploy only on tags
REM  )
@echo off
