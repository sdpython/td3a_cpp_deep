@echo off
set current=%~dp0
set root=%current%..
cd %root%
@echo ##################
@echo Compile
@echo ##################
set pythonexe=python

@echo running %root%\setup.py build_ext --inplace
%pythonexe% -u %root%\setup.py build_ext --inplace

if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Compile.
@echo ##################
cd %root%
@echo Build
@echo running setup.py bdist_wheel sdist
@echo ##################
%pythonexe% -u setup.py bdist_wheel sdist
if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Build.
cd %current%