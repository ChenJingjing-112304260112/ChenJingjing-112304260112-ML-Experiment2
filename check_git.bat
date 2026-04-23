@echo off

:: 检查git是否可用
git --version

if %errorlevel% equ 0 (
    echo Git is available
) else (
    echo Git is not available
)

pause