@echo off
echo 开始自动上传到GitHub并启动监控...

rem 进入项目目录
cd /d "c:\Users\51273\Desktop\机器学习2"

rem 检查Git状态
echo 检查Git状态...
git status

rem 添加所有文件
echo 添加所有文件...
git add .

rem 提交更改
echo 提交更改...
git commit -m "auto-update: initial upload"

rem 推送到GitHub
echo 推送到GitHub...
git push

rem 启动监控脚本
echo 启动监控脚本...
start powershell -ExecutionPolicy Bypass -File monitor_github.ps1

echo 自动上传和监控已启动！
echo 按任意键退出...
pause > nul