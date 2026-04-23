# 切换到项目目录
Set-Location "c:\Users\51273\Desktop\机器学习2"

# 检查是否有未提交的更改
Write-Host "检查状态..."
git status

# 添加所有更改
Write-Host "添加更改..."
git add .

# 提交更改
Write-Host "提交更改..."
git commit -m "Update submission.csv with improved predictions"

# 推送到GitHub
Write-Host "推送到GitHub..."
git push origin main

Write-Host "自动上传完成！"
Read-Host "按 Enter 键退出..."