@echo off
REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing pip...
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    del get-pip.py
) else (
    echo Pip is already installed.
)

REM Check and update pip if necessary
echo Checking pip version...
pip install --upgrade pip

REM Check and install/update necessary packages
echo Checking required packages...

REM Check and install/update statistics
pip show statistics >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing statistics...
    pip install statistics
) else (
    echo statistics is already installed.
)
pip install --upgrade statistics

REM Check and install/update pyquaternion
pip show pyquaternion >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing pyquaternion...
    pip install pyquaternion
) else (
    echo pyquaternion is already installed.
)
pip install --upgrade pyquaternion

REM Check and install/update numpy
pip show numpy >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing numpy...
    pip install numpy
) else (
    echo numpy is already installed.
)
pip install --upgrade numpy

REM Run the main script
echo Running calculator.py...
python calculator.py

REM End of script
pause
