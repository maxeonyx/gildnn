# loop.ps1
# Outer loop for the gildnn project.
# Launches OpenCode in a persistent session. If it exits for any reason, relaunches it.
# OpenCode is the real working environment — this is just the restart wrapper.

$sessionId = "ses_1fabddbe7ffe56KcM6SP73W55s"

$prompt = @"
This is an autonomous ML research project. You should continue working for as long as possible. The user is not here — do not ask questions, do not wait for input, do not exit to wait for anything.

Read PLAN.md and any TASK-*.ignore.md files in the root to orient on current state. Read PROCESS.md for how work is done here.

You may have stopped for any reason — error, network issue, task completion, reboot. If you know why, great. If not, read the current state and continue or pick up the next task.

Check the time. If it is after 4pm and no daily report exists for today (research/daily/YYYY-MM-DD.md), write it at the next natural stopping point. If it is also Thursday, write the weekly report too.

Priority order:
1. Improve the process — if docs, files, or the process itself are wrong or unclear, fix them first
2. Reports — write any that are due
3. Clean up — not just code, everything in the repo: docs, question folders, experiment records, these files. If something is wrong, fix it.
4. Integrate — finished experimental code into core/, findings into question folders
5. Continue existing experiments
6. Start new experiments

If a training run is active in the background: do not wait for it. Continue other work (cleanup, reports, integration). Do not start a second training run while one is active.

Work in small verified steps. Record what you find as you go so the eventual write-up is easy.
"@

Write-Host "Starting gildnn loop. Press Ctrl+C to stop."
Write-Host ""

$crashCount = 0
$lastLaunch = Get-Date

while ($true) {
    $now = Get-Date
    Write-Host "[$($now.ToString('yyyy-MM-dd HH:mm:ss'))] Launching OpenCode (session $sessionId)..."

    opencode run --session $sessionId --no-ephemeral $prompt

    $exitCode = $LASTEXITCODE
    $elapsed = ((Get-Date) - $lastLaunch).TotalSeconds
    $lastLaunch = Get-Date
    Write-Host "[$($lastLaunch.ToString('yyyy-MM-dd HH:mm:ss'))] OpenCode exited (code $exitCode, ran ${elapsed}s)"

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
