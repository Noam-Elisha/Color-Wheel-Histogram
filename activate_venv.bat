@echo off
REM Simple batch script to activate the virtual environment
REM Usage: activate_venv.bat

echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Keep the command prompt open with the activated environment
cmd /k