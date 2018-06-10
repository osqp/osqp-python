@echo on


:: Perform Python tests
:: -------------------------------------------------------
:: Install python interface
cd %APPVEYOR_BUILD_FOLDER%
python setup.py install

:: Test python interface
cd %APPVEYOR_BUILD_FOLDER%
python -m pytest
if errorlevel 1 exit /b 1

@echo off
