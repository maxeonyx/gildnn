<#
.SYNOPSIS
    Quick visibility into what the autonomous research loop is doing.
.DESCRIPTION
    Shows: active experiments, recent commits, experiment output, and any pending dictation checks.
    Run from the repo root: .\loop-status.ps1
#>
param(
    [int]$CommitCount = 5,
    [int]$LogLines = 10
)

$repoRoot = $PSScriptRoot
Set-Location $repoRoot

Write-Host "`n=== GILDNN Loop Status ===" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n"

# Active experiment
if (Test-Path runs/active.lock) {
    $lock = Get-Content runs/active.lock -Raw
    Write-Host "ACTIVE RUN:" -ForegroundColor Yellow
    Write-Host "  $lock"
} else {
    Write-Host "No active run (runs/active.lock missing)" -ForegroundColor DarkGray
}

# Find experiment logs
$logs = Get-ChildItem runs/*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($logs) {
    Write-Host "`nLATEST LOG: $($logs[0].Name) (modified $(($logs[0].LastWriteTime).ToString('HH:mm:ss')))" -ForegroundColor Yellow
    $content = Get-Content $logs[0].FullName -Tail $LogLines -ErrorAction SilentlyContinue
    if ($content) {
        $content | ForEach-Object { Write-Host "  $_" }
    } else {
        Write-Host "  (empty)" -ForegroundColor DarkGray
    }
}

# Recent commits
Write-Host "`nRECENT COMMITS:" -ForegroundColor Yellow
git log --oneline -n $CommitCount --format="  %h %s (%ar)" 2>$null

# New dictations
Write-Host "`nDICTATION CHECK:" -ForegroundColor Yellow
$check = & .\.venv\Scripts\python.exe -m core.check_dictations 2>&1
Write-Host "  $check"

# PLAN.md current item
Write-Host "`nCURRENT PLAN:" -ForegroundColor Yellow
$planNow = Get-Content PLAN.md | Where-Object { $_ -match '^\- \[ \]' } | Select-Object -First 3
$planNow | ForEach-Object { Write-Host "  $_" }

Write-Host ""
