# loop.ps1 — gildnn autonomous loop manager
#
# HOW IT WORKS:
# This script uses Windows Task Scheduler to detach the loop from your
# terminal. When you run it, it registers a scheduled task that calls back
# into this same script with the hidden "run" action. That second invocation
# is the actual forever-loop (launch OpenCode, wait, relaunch). Because Task
# Scheduler owns that process, closing your terminal/SSH/tmux won't kill it.
#
# The task has no triggers — it doesn't run on boot or login. It only runs
# when you start it. It re-registers the same task name each time (no clutter).
#
# USAGE:
#   .\loop.ps1                  — start if needed, show status, tail logs
#   .\loop.ps1 stop             — stop the loop
#   .\loop.ps1 -DummyCommand    — test mode (fake iterations, separate runtime dir)

param(
    [Parameter(Position = 0)]
    [string]$Action = '',
    [switch]$DummyCommand,
    [int]$MaxIterations = 0
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$sessionId = 'ses_1c007665effen45IK4eRb2NBcn'
$runtimeName = if ($DummyCommand) { 'loop-dummy.ignore' } else { 'loop-runtime.ignore' }
$runtimeDir = Join-Path $PSScriptRoot 'runs' $runtimeName
$stdoutLog = Join-Path $runtimeDir 'stdout.log'
$taskName = if ($DummyCommand) { 'gildnn-loop-dummy' } else { 'gildnn-loop' }
$pidFile = Join-Path $runtimeDir 'loop.pid'

function Get-LoopProcess {
    if (-not (Test-Path -LiteralPath $pidFile)) { return $null }
    $pidText = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if (-not $pidText) { return $null }
    $loopPid = 0
    if (-not [int]::TryParse($pidText, [ref]$loopPid)) { return $null }
    try { return Get-Process -Id $loopPid -ErrorAction Stop }
    catch { return $null }
}

function Ensure-LoopTask {
    # Skip re-registration if the task already exists (avoids needing elevation)
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -ne $existing) { return }

    $pwsh = (Get-Command pwsh).Source
    $arguments = @(
        '-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden',
        '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"",
        'run'
    )
    if ($DummyCommand) { $arguments += '-DummyCommand' }
    if ($MaxIterations -gt 0) { $arguments += '-MaxIterations'; $arguments += $MaxIterations }

    $action = New-ScheduledTaskAction -Execute $pwsh -Argument ($arguments -join ' ')
    $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
    $task = New-ScheduledTask -Action $action -Principal $principal -Settings $settings
    Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
}

function Start-Loop {
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    $existing = Get-LoopProcess
    if ($null -ne $existing) {
        Write-Output "Loop already running (PID $($existing.Id))."
        return
    }
    Ensure-LoopTask
    Start-ScheduledTask -TaskName $taskName
    $deadline = (Get-Date).AddSeconds(10)
    do {
        Start-Sleep -Milliseconds 250
        $proc = Get-LoopProcess
    } while ($null -eq $proc -and (Get-Date) -lt $deadline)
    if ($null -ne $proc) { Write-Output "Started (PID $($proc.Id))." }
    else { Write-Output "Task started — waiting for PID..." }
}

function Stop-Loop {
    $proc = Get-LoopProcess
    if ($null -eq $proc) { Write-Output 'Loop is not running.'; return }
    Stop-Process -Id $proc.Id -Force
    Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
    try { Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue } catch {}
    Write-Output "Stopped loop wrapper (PID $($proc.Id))."

    # Kill any orphaned opencode process running with this loop's session ID
    $orphans = Get-CimInstance Win32_Process -Filter "Name='opencode.exe'" |
        Where-Object { $_.CommandLine -match [regex]::Escape($sessionId) }
    foreach ($o in $orphans) {
        Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Output "Stopped orphaned opencode (PID $($o.ProcessId))."
    }
}

function Follow-Logs {
    if (-not (Test-Path -LiteralPath $stdoutLog)) {
        New-Item -ItemType File -Path $stdoutLog -Force | Out-Null
    }
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Get-Content -LiteralPath $stdoutLog -Tail 80 -Wait -Encoding utf8
}

# --- The forever-loop (invoked by the scheduled task, not by the user) ---

function Run-Loop {
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    Set-Content -LiteralPath $pidFile -Value $PID
    $null = New-Item -ItemType File -Path $stdoutLog -Force

    function Log($msg) { $msg | Out-File -LiteralPath $stdoutLog -Append -Encoding utf8 }

    try {
        Log "Loop started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        $crashCount = 0
        $iteration = 0
        $lastLaunch = Get-Date
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

        while ($true) {
            $iteration++
            $mode = if ($DummyCommand) { 'dummy' } else { 'OpenCode' }
            Log "[$(Get-Date -Format 'HH:mm:ss')] iteration $iteration — launching $mode"

            $prompt = Get-Content -Path "$PSScriptRoot\loop-prompt.md" -Raw

            if ($DummyCommand) {
                & pwsh -NoProfile -Command "Write-Output ('dummy iteration ' + $iteration); exit 0" 2>&1 | Out-File -LiteralPath $stdoutLog -Append -Encoding utf8
                $exitCode = $LASTEXITCODE
            }
            else {
                opencode run --agent arrange --model github-copilot-max/claude-opus-4.6 --variant high --session $sessionId $prompt 2>&1 | Out-File -LiteralPath $stdoutLog -Append -Encoding utf8
                $exitCode = $LASTEXITCODE
            }

            $elapsed = ((Get-Date) - $lastLaunch).TotalSeconds
            $lastLaunch = Get-Date
            Log "[$(Get-Date -Format 'HH:mm:ss')] exited (code $exitCode, ${elapsed}s)"

            if ($MaxIterations -gt 0 -and $iteration -ge $MaxIterations) {
                Log "Reached max iterations ($MaxIterations). Done."
                break
            }

            if ($elapsed -lt 30) {
                $crashCount++
                $backoff = if ($DummyCommand) { 1 } else { [Math]::Min(60 * $crashCount, 300) }
                Log "Fast exit #$crashCount — waiting ${backoff}s"
                Start-Sleep -Seconds $backoff
            }
            else {
                $crashCount = 0
                Log 'Waiting 10s...'
                Start-Sleep -Seconds 10
            }
        }
    }
    finally {
        Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
    }
}

# --- Entry point ---

switch ($Action.ToLowerInvariant()) {
    'stop' { Stop-Loop }
    'run'  { Run-Loop }
    default {
        Start-Loop
        $proc = Get-LoopProcess
        if ($null -eq $proc) { Write-Output 'Loop: not running' }
        else { Write-Output "Loop: running (PID $($proc.Id), started $($proc.StartTime.ToString('yyyy-MM-dd HH:mm:ss')))" }
        Write-Output "Log:  $stdoutLog"
        Write-Output ''
        Follow-Logs
    }
}
