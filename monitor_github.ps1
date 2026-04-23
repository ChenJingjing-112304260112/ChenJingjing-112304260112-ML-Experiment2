$projectPath = "c:\Users\51273\Desktop\机器学习2"
$lastCheck = Get-Date

Write-Host "Start monitoring file changes, checking every 30 seconds..."
Write-Host "Press Ctrl+C to stop monitoring"

while ($true) {
    # Check for file changes
    $changedFiles = Get-ChildItem -Path $projectPath -Recurse | Where-Object {
        $_.LastWriteTime -gt $lastCheck -and $_.Name -notlike "*.git*"
    }
    
    if ($changedFiles.Count -gt 0) {
        Write-Host "Detected file changes, updating GitHub..."
        Set-Location -Path $projectPath
        
        # Execute Git commands
        try {
            git add .
            git commit -m "auto-update: $(Get-Date)"
            git push
            
            $lastCheck = Get-Date
            Write-Host "Update completed!"
        } catch {
            Write-Host "Update failed: $_"
        }
    }
    
    # Check every 30 seconds
    Start-Sleep -Seconds 30
}