@echo off
echo 启动GitHub自动监控...
powershell -ExecutionPolicy Bypass -File monitor_github.ps1
pause