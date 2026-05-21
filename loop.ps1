# loop.ps1
# Run this with no arguments.
#
# User-facing behavior:
# - if the loop is not running, start it in the background
# - show the current status
# - tail the loop log
#
# Agent note:
# - do not change the no-arg behavior above
# - test loop changes with the dummy-command path, not Max's real loop

param(
    [string]$Action = 'console',

    [Alias('h')]
    [switch]$Help,

    [switch]$DummyCommand,

    [int]$MaxIterations = 0,

    [string]$RuntimeName = 'loop-runtime.ignore'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$sessionId = 'ses_1c007665effen45IK4eRb2NBcn'
$runtimeDir = Join-Path $PSScriptRoot (Join-Path 'runs' $RuntimeName)
$stdoutLog = Join-Path $runtimeDir 'stdout.log'
$stderrLog = Join-Path $runtimeDir 'stderr.log'
$pidFile = Join-Path $runtimeDir 'loop.pid'
$metaFile = Join-Path $runtimeDir 'loop-state.json'

function Show-Help {
    Write-Output 'Usage:'
    Write-Output '  .\loop.ps1'
    Write-Output '  .\loop.ps1 help'
    Write-Output '  .\loop.ps1 --help'
    Write-Output '  .\loop.ps1 <action>'
    Write-Output ''
    Write-Output 'Default behavior with no arguments:'
    Write-Output '  - start the loop if it is not running'
    Write-Output '  - show the current status'
    Write-Output '  - tail the loop log'
    Write-Output ''
    Write-Output 'Actions:'
    Write-Output '  start    Start the loop in the background if needed'
    Write-Output '  status   Show whether the loop is running'
    Write-Output '  logs     Tail the loop stdout log'
    Write-Output '  stop     Stop the loop if it is running'
    Write-Output '  restart  Restart the loop'
    Write-Output '  help     Show this help'
    Write-Output ''
    Write-Output 'Agent/test options:'
    Write-Output '  -DummyCommand              Run a dummy command instead of OpenCode'
    Write-Output '  -MaxIterations <n>         Stop after n iterations'
    Write-Output '  -RuntimeName <name>        Use a separate .ignore runtime dir under runs/'
}

function Ensure-RuntimeDir {
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
}

function Get-LoopProcess {
    if (-not (Test-Path -LiteralPath $pidFile)) {
        return $null
    }

    $pidText = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if (-not $pidText) {
        return $null
    }

    $loopPid = 0
    if (-not [int]::TryParse($pidText, [ref]$loopPid)) {
        return $null
    }

    try {
        return Get-Process -Id $loopPid -ErrorAction Stop
    }
    catch {
        return $null
    }
}

function Write-StateFile {
    param(
        [int]$LoopPid
    )

    $state = [pscustomobject]@{
        pid = $LoopPid
        sessionId = $sessionId
        startedAt = (Get-Date).ToString('o')
        repoRoot = $PSScriptRoot
        stdoutLog = $stdoutLog
        stderrLog = $stderrLog
    }
    $state | ConvertTo-Json | Set-Content -LiteralPath $metaFile
}

function Show-Status {
    $proc = Get-LoopProcess
    if ($null -eq $proc) {
        Write-Output 'Loop status: not running'
        Write-Output "Repo: $PSScriptRoot"
        Write-Output "Stdout log: $stdoutLog"
        Write-Output "Stderr log: $stderrLog"
        return
    }

    Write-Output 'Loop status: running'
    Write-Output "PID: $($proc.Id)"
    Write-Output "Started: $($proc.StartTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Output "Session: $sessionId"
    Write-Output "Repo: $PSScriptRoot"
    Write-Output "Stdout log: $stdoutLog"
    Write-Output "Stderr log: $stderrLog"
}

function Start-Loop {
    Ensure-RuntimeDir

    $existing = Get-LoopProcess
    if ($null -ne $existing) {
        Write-Output "Loop already running (PID $($existing.Id))."
        Show-Status
        return
    }

    $pwsh = (Get-Command pwsh).Source
    $argumentList = @(
        '-NoProfile'
        '-ExecutionPolicy'
        'Bypass'
        '-File'
        $PSCommandPath
        '-Action'
        'run'
        '-RuntimeName'
        $RuntimeName
    )
    if ($DummyCommand) {
        $argumentList += '-DummyCommand'
    }
    if ($MaxIterations -gt 0) {
        $argumentList += '-MaxIterations'
        $argumentList += $MaxIterations
    }
    $proc = Start-Process -FilePath $pwsh -ArgumentList $argumentList -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
    Set-Content -LiteralPath $pidFile -Value $proc.Id
    Write-StateFile -LoopPid $proc.Id

    Write-Output "Started loop in background (PID $($proc.Id))."
    Write-Output "Stdout log: $stdoutLog"
    Write-Output "Stderr log: $stderrLog"
}

function Stop-Loop {
    $proc = Get-LoopProcess
    if ($null -eq $proc) {
        Write-Output 'Loop is not running.'
        return
    }

    Stop-Process -Id $proc.Id
    Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
    Write-Output "Stopped loop process $($proc.Id)."
}

function Follow-Logs {
    Ensure-RuntimeDir

    if (-not (Test-Path -LiteralPath $stdoutLog)) {
        New-Item -ItemType File -Path $stdoutLog -Force | Out-Null
    }

    Write-Output "Following $stdoutLog"
    if ((Test-Path -LiteralPath $stderrLog) -and (Get-Item -LiteralPath $stderrLog).Length -gt 0) {
        Write-Output "Note: stderr also has output: $stderrLog"
    }
    Get-Content -LiteralPath $stdoutLog -Tail 80 -Wait
}

function Enter-ConsoleMode {
    $proc = Get-LoopProcess
    if ($null -eq $proc) {
        Start-Loop
        Start-Sleep -Seconds 1
    }

    Show-Status
    Write-Output ''
    Follow-Logs
}

function Run-Loop {
    Ensure-RuntimeDir

    Write-Output 'Starting gildnn loop in background. Press Ctrl+C in the child process to stop.'
    Write-Output ''

    $crashCount = 0
    $iteration = 0
    $lastLaunch = Get-Date

    while ($true) {
        $iteration++
        $now = Get-Date
        $mode = if ($DummyCommand) { 'dummy command' } else { 'OpenCode' }
        Write-Output "[$($now.ToString('yyyy-MM-dd HH:mm:ss'))] Launching $mode iteration $iteration (session $sessionId)..."

        $prompt = Get-Content -Path "$PSScriptRoot\loop-prompt.md" -Raw
        Write-Output $prompt

        if ($DummyCommand) {
            & pwsh -NoProfile -Command "Write-Output ('dummy iteration ' + $iteration); exit 0"
            $exitCode = $LASTEXITCODE
        }
        else {
            opencode run --agent arrange --model github-copilot-max/claude-opus-4.6 --variant high --session $sessionId $prompt
            $exitCode = $LASTEXITCODE
        }

        $elapsed = ((Get-Date) - $lastLaunch).TotalSeconds
        $lastLaunch = Get-Date
        Write-Output "[$($lastLaunch.ToString('yyyy-MM-dd HH:mm:ss'))] $mode exited (code $exitCode, ran ${elapsed}s, iteration $iteration)"

        if ($MaxIterations -gt 0 -and $iteration -ge $MaxIterations) {
            Write-Output "Reached max iterations ($MaxIterations). Loop finished."
            break
        }

        if ($elapsed -lt 30) {
            $crashCount++
            if ($DummyCommand) {
                $backoff = 1
            }
            else {
                $backoff = [Math]::Min(60 * $crashCount, 300)
            }
            Write-Output "Fast exit #$crashCount. Waiting ${backoff}s before relaunch..."
            Start-Sleep -Seconds $backoff
        }
        else {
            $crashCount = 0
            Write-Output 'Waiting 10s before relaunch...'
            Start-Sleep -Seconds 10
        }
    }
}

$normalizedAction = $Action.ToLowerInvariant()

if ($Help -or $normalizedAction -eq 'help' -or $Action -eq '--help') {
    Show-Help
    return
}

switch ($normalizedAction) {
    'console' { Enter-ConsoleMode }
    'start' { Start-Loop }
    'run' { Run-Loop }
    'status' { Show-Status }
    'logs' { Follow-Logs }
    'stop' { Stop-Loop }
    'restart' {
        Stop-Loop
        Start-Loop
    }
    default {
        throw "Unknown action '$Action'. Run .\loop.ps1 --help for usage."
    }
}
