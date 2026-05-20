# loop.ps1
# Outer loop for the gildnn project.
# Launches OpenCode in a persistent session. If it exits for any reason, relaunches it.
# OpenCode is the real working environment — this is just the restart wrapper.

# Ensure we're always running from the repo root, regardless of where the script was launched from
Set-Location $PSScriptRoot

$sessionId = "ses_1c007665effen45IK4eRb2NBcn"

Write-Host "Starting gildnn loop. Press Ctrl+C to stop."
Write-Host ""

$maxIterations = 10
$crashCount = 0
$iteration = 0
$lastLaunch = Get-Date

while ($iteration -lt $maxIterations) {
    $iteration++
    $now = Get-Date
    Write-Host "[$($now.ToString('yyyy-MM-dd HH:mm:ss'))] Launching OpenCode iteration $iteration/$maxIterations (session $sessionId)..."

    $prompt = Get-Content -Path "$PSScriptRoot\loop-prompt.md" -Raw
    $prompt = "$prompt $($args[0])"
    echo $prompt
    opencode run --agent arrange --model github-copilot-max/claude-opus-4.6 --variant high --session $sessionId $prompt
    
    $exitCode = $LASTEXITCODE
    $elapsed = ((Get-Date) - $lastLaunch).TotalSeconds
    $lastLaunch = Get-Date
    Write-Host "[$($lastLaunch.ToString('yyyy-MM-dd HH:mm:ss'))] OpenCode exited (code $exitCode, ran ${elapsed}s, iteration $iteration/$maxIterations)"

    # Back off if it's crashing fast, to avoid a tight crash loop
    if ($elapsed -lt 30) {
        $crashCount++
        $backoff = [Math]::Min(60 * $crashCount, 300)
        Write-Host "Fast exit #$crashCount — waiting ${backoff}s before relaunch..."
        Start-Sleep -Seconds $backoff
    } else {
        $crashCount = 0
        Write-Host "Waiting 10s before relaunch..."
        Start-Sleep -Seconds 10
    }
}

Write-Host ""
Write-Host "Completed $maxIterations iterations. Loop finished."
