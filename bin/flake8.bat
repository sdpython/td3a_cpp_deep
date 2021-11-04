@echo off
set current=%~dp0
set root=%current%..
cd %root%
set pythonexe=python

%pythonexe% -m autopep8 -r --in-place td3a_cpp_deep
%pythonexe% -m autopep8 -r --in-place examples
%pythonexe% -m autopep8 -r --in-place tests

@echo running 'python -m flake8 td3a_cpp_deep tests examples'
%pythonexe% -m flake8 td3a_cpp_deep tests examples setup.py doc/conf.py

if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Testing.