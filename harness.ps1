# harness.ps1
# Outer loop for the gildnn autonomous research project.
# Launches OpenCode repeatedly. If OpenCode exits for any reason, relaunches it.
# This script should be refined as the process matures.

$prompt = @"
You are picking up an autonomous ML research session for the gildnn project.

Read these files before doing anything else:
1. PLAN.md — current state and next task
2. Any TASK-*.ignore.md files in the root — mid-task context from last session
3. VISION.md briefly — to reorient on goals

Then act. Do not ask questions. The user is not here.

--- Situations ---

If a training run is currently active:
  Sleep (Start-Sleep -Seconds 300) and check again. Do NOT exit to wait.

If you are mid-task with clear next steps:
  Continue. Pick up exactly where the last session left off.

If a task just completed:
  Update PLAN.md to reflect completion. Pick the next task.

If something is broken or unclear:
  Clean it up. Read the dictations (dictations/) if you are unsure of intent.
  Redo from scratch is always an option — sometimes cleaner than fixing a tangle.

--- Priority order ---

When choosing what to work on, in order:
1. Clean up anything broken or half-finished
2. Make the codebase smaller — delete, simplify, consolidate
3. Refactor the process (PROCESS.md, AGENTS.md, harness.ps1)
4. Refactor the repo — integrate experimental code into core/
5. Continue existing experiments
6. Start new experiments

Integrating finished experimental code is higher priority than starting new experiments.

--- Rules ---

- Do not exit to "wait" for a training run. Sleep instead.
- Do not ask the user anything. They are not here.
- If you cannot complete a task in this session: say so clearly, leave a clean TASK-*.ignore.md handoff, update PLAN.md. Do not deliver something broken.
- Keep the codebase small. This is an explicit quality metric.
- Do not modify anything in dictations/.
- Open questions stay open. Do not state architectural decisions as settled unless experimentally confirmed.

--- If this is your first session ---

No code exists yet. Read PLAN.md for immediate priorities.
The first task is environment setup (Python + JAX + CUDA), then a sanity-check experiment.
"@

Write-Host "Starting gildnn harness. Press Ctrl+C to stop."
Write-Host ""

while ($true) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Launching OpenCode..."
    
    # Launch OpenCode with the reorientation prompt
    # Adjust the opencode command/path as needed for your installation
    opencode --message $prompt
    
    $exitCode = $LASTEXITCODE
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] OpenCode exited with code $exitCode"
    
    # Brief pause before relaunch to avoid tight loop on crash
    Write-Host "Waiting 10 seconds before relaunch..."
    Start-Sleep -Seconds 10
}
