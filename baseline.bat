@echo off
setlocal enabledelayedexpansion

REM 定义数据目录
set "data_dir=data"

REM 检查数据目录是否存在
if not exist "%data_dir%" (
    echo Data directory does not exist.
    exit /b 1
)

REM 遍历 data 目录下的所有子目录
for /d %%d in ("%data_dir%\*") do (
    set "relative_data_dir=%%~nxd"
    set "full_data_prefix=data\!relative_data_dir!"
    echo Processing directory: !full_data_prefix!

    REM 执行 Python 脚本
    python main.py --data_prefix "!full_data_prefix!" --train_config "./train_config/config.yml" --repeat_time 1
)

echo All directories processed.
pause