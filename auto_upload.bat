@echo off
setlocal

:: 切换到项目目录
cd /d c:\Users\51273\Desktop\机器学习2

:: 检查是否有未提交的更改
git status

:: 添加所有更改
git add .

:: 提交更改
git commit -m "Update submission.csv with improved predictions"

:: 推送到GitHub
git push origin main

echo 自动上传完成！
pause