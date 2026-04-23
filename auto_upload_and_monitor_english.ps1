Write-Host "Starting auto upload to GitHub and monitoring..."

# Navigate to project directory
Set-Location -Path "c:\Users\51273\Desktop\机器学习2"

# Check Git status
Write-Host "Checking Git status..."
git status

# Add all files
Write-Host "Adding all files..."
git add .

# Commit changes
Write-Host "Committing changes..."
git commit -m "auto-update: initial upload"

# Push to GitHub
Write-Host "Pushing to GitHub..."
git push

# Start monitoring script
Write-Host "Starting monitoring script..."
Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File monitor_github.ps1"

Write-Host "Auto upload and monitoring started!"
Write-Host "Press any key to exit..."
$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown') | Out-Null