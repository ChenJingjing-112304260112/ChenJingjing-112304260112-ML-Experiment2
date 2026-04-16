@echo off

echo ==== Updating GitHub repository ====

rem 检查Git状态
echo Checking Git status...
git status

rem 添加所有文件
echo Adding all files...
git add .

rem 提交更改
echo Committing changes...
git commit -m "auto-update: latest changes"

rem 推送到GitHub
echo Pushing to GitHub...
git push

echo ==== Update completed ====
pause