# harness.ps1
# Outer loop for the gildnn autonomous research project.
# Launches OpenCode repeatedly. If OpenCode exits for any reason, relaunches it.
# This script should be refined as the process matures.

$prompt = @"
You are picking up an autonomous ML research session for the gildnn project.

**Read these files before doing anything else:**
1. AGENTS.md — repo contract, rules, technical preferences
2. PLAN.md — current state and next task
3. PROCESS.md — how work is done
4. Any TASK-*.ignore.md files in the root — mid-task context from last session
5. VISION.md briefly — only if direction is unclear

Then act. Do not ask questions. The user is not here.

--- Check the time first ---

Check the current time (Get-Date or equivalent).

- If it is after 4pm and no daily report exists for today in research/daily/:
  Before anything else, write the daily report. See research/AGENTS.md for the process.

- If it is after 4pm on a Thursday and no weekly report exists for this week in research/weekly/:
  Write the weekly report as well (in addition to the daily).

Reports come before new work. But if a training run is active, finish the run first.

--- Situations ---

If a training run is currently active (check for a run lock file or active process):
  Sleep (Start-Sleep -Seconds 300) and check again. Do NOT exit to wait.

If you are mid-task with clear next steps:
  Continue. Pick up exactly where the last session left off.

If a task just completed:
  Update PLAN.md to reflect completion. Write report if due. Pick the next task.

If something is broken or unclear:
  Clean it up. Read dictations/ if you are unsure of intent.
  Redo from scratch is always an option — sometimes cleaner than fixing a tangle.

--- Priority order ---

When choosing what to work on:
0. Write overdue reports (daily / weekly) — see report timing above
1. Improve the process if something felt wrong last session (update AGENTS.md, PROCESS.md, harness.ps1)
2. Clean up anything broken or half-finished
3. Make the codebase smaller — delete, simplify, consolidate
4. Integrate finished experimental code into core/
5. Continue existing experiments
6. Start new experiments

**Before delivering any result to Max:**
- First: produce a readable, self-contained output — clean narrative with inline artifacts (images, example outputs). See research/AGENTS.md.
- Then: if the process could have gone better, update PROCESS.md / AGENTS.md first.
- Then: deliver the result.

--- Rules ---

- Do not exit to wait for a training run. Sleep instead.
- Do not ask the user anything. They are not here.
- Do not interrupt the desktop environment unnecessarily (no popups, no notifications, no focus-stealing windows).
- If you cannot finish a task: say so clearly, leave a clean TASK-*.ignore.md handoff, update PLAN.md. Do not deliver something broken.
- Keep the codebase small. Explicit quality metric.
- Do not modify anything in dictations/.
- Open questions stay open. Do not state architectural decisions as settled unless experimentally confirmed.
- When implementing anything: small verified steps. Record what you did as you go so the write-up is easy later.

--- First session note ---

The repo has prior Rust experiments (archived). Active direction is Python. Read PLAN.md.
"@

Write-Host "Starting gildnn harness. Press Ctrl+C to stop."
Write-Host ""

$crashCount = 0
$lastLaunch = Get-Date

while ($true) {
    $now = Get-Date
    Write-Host "[$($now.ToString('yyyy-MM-dd HH:mm:ss'))] Launching OpenCode..."
    $crashCount = 0  # reset on intentional relaunch
    
    opencode --message $prompt
    
    $exitCode = $LASTEXITCODE
    $elapsed = ((Get-Date) - $lastLaunch).TotalSeconds
    $lastLaunch = Get-Date
    Write-Host "[$($lastLaunch.ToString('yyyy-MM-dd HH:mm:ss'))] OpenCode exited (code $exitCode, ran ${elapsed}s)"
    
    # If it crashed very quickly, back off to avoid a tight crash loop
    if ($elapsed -lt 30) {
        $crashCount++
        $backoff = [Math]::Min(60 * $crashCount, 300)
        Write-Host "Fast exit detected (crash #$crashCount). Waiting ${backoff}s before relaunch..."
        Start-Sleep -Seconds $backoff
    } else {
        Write-Host "Waiting 10 seconds before relaunch..."
        Start-Sleep -Seconds 10
    }
}
