$projectPath = "c:\Users\51273\Desktop\机器学习2"
$lastCheck = Get-Date

Write-Host "开始监控文件变化，每30秒检查一次..."
Write-Host "按 Ctrl+C 停止监控"

while ($true) {
    # 检查文件变化
    $changedFiles = Get-ChildItem -Path $projectPath -Recurse | Where-Object {
        $_.LastWriteTime -gt $lastCheck -and $_.Name -notlike "*.git*"
    }
    
    if ($changedFiles.Count -gt 0) {
        Write-Host "检测到文件变化，更新GitHub..."
        Set-Location -Path $projectPath
        
        # 执行Git命令
        try {
            git add .
            git commit -m "auto-update: $(Get-Date)"
            git push
            
            $lastCheck = Get-Date
            Write-Host "更新完成！"
        } catch {
            Write-Host "更新失败: $_"
        }
    }
    
    # 每30秒检查一次
    Start-Sleep -Seconds 30
}