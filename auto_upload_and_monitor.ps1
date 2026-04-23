Write-Host "开始自动上传到GitHub并启动监控..."

# 进入项目目录
Set-Location -Path "c:\Users\51273\Desktop\机器学习2"

# 检查Git状态
Write-Host "检查Git状态..."
git status

# 添加所有文件
Write-Host "添加所有文件..."
git add .

# 提交更改
Write-Host "提交更改..."
git commit -m "auto-update: initial upload"

# 推送到GitHub
Write-Host "推送到GitHub..."
git push

# 启动监控脚本
Write-Host "启动监控脚本..."
Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File monitor_github.ps1"

Write-Host "自动上传和监控已启动！"
Write-Host "按任意键退出..."
$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown') | Out-Null