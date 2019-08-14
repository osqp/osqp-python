@echo on

@setlocal enabledelayedexpansion

IF "%DISTRIB%"=="conda" (

REM Build and test conda recipe
conda build --python %PYTHON_VERSION% conda-recipe --output-folder conda-bld

)


IF "%DISTRIB%"=="pip" (

REM Creating pip wheel
python setup.py bdist_wheel

)
