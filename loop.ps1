# loop.ps1
# Outer loop for the gildnn project.
# Launches OpenCode in a persistent session. If it exits for any reason, relaunches it.
# OpenCode is the real working environment — this is just the restart wrapper.

# Ensure we're always running from the repo root, regardless of where the script was launched from
Set-Location $PSScriptRoot

$sessionId = "ses_1f9625133ffelr3Li5SkSp02ET"

$prompt = @"
This is an autonomous personal ML research project. Keep working until there is a good reason to stop. The user is usually not here: do not wait for input, do not ask questions unless absolutely necessary, and do not exit just because one step finished.

Reorient fast:
- Read PLAN.md and any TASK-*.ignore.md files in the repo root.
- Read PROCESS.md if you need the working rules.
- Check the time and write any due daily/weekly report at the next natural stopping point.
- If runs/active.lock exists, do not wait for the run; do other useful work and do not start a second long run.

Your job is to do research, not bureaucracy.

Prefer:
1. fixing broken or misleading process/docs
2. writing any due reports
3. cleaning up half-finished or confusing work
4. integrating finished experimental code into core/
5. continuing an existing experiment
6. starting a new small experiment

When choosing what to do next, pick the cheapest honest step that could produce useful evidence. Keep scope small. Save artifacts. Inspect actual outputs. Open questions stay open.

The experiment ladder is mandatory:
- overfit one batch
- tiny model / tiny data
- inspect outputs
- scale gradually
- change one thing at a time

Redo from scratch is allowed. Deleting clutter is good. A small clear result is better than a grand blocked plan.
"@

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

    opencode run --agent arrange --model github-copilot-max/gpt-5.4 --session $sessionId --no-ephemeral=true $prompt  # NOTE: in newer versions of the opencode fork, --no-ephemeral may be removed (headless sessions stored by default). If this flag breaks, just remove it.

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
