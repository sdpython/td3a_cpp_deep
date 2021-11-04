@echo off
set current=%~dp0
set root=%current%..
cd %root%
set pythonexe=python

@echo running 'python -m sphinx -T -b html doc dist/html'
%pythonexe% -m sphinx -T -b html doc dist/html

if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Testing.