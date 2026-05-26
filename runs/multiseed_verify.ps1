#!/usr/bin/env pwsh
# Multi-seed verification: recurrent_4_d256 vs distinct_4_d128 at 900K chars
# Runs 3 seeds for each configuration to get confidence intervals
# Expected total runtime: ~87 min (6 experiments × ~14 min average)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$python = ".\.venv\Scripts\python.exe"
$seeds = @(42, 137, 7)
$artifactDir = "experiments\tinyshakespeare\artifacts\recurrent_depth_lm"

$results = @()
$startTime = Get-Date

Write-Host "=== Multi-seed verification started at $(Get-Date -Format 'HH:mm:ss') ==="
Write-Host "Running $($seeds.Count) seeds × 2 configs = $($seeds.Count * 2) experiments"
Write-Host ""

foreach ($seed in $seeds) {
    # --- d=256 run (primary: recurrent_4_d256) ---
    $reportWide = "$artifactDir\verify_wide_seed${seed}.json"
    $logWide = "runs\verify_wide_seed${seed}.log"
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting d=256, seed=$seed..."
    
    & $python -m runs.recurrent_depth_lm `
        --seed $seed `
        --d-model 256 --n-heads 8 --ff-dim 1024 `
        --train-characters 900000 --val-characters 100000 `
        --report-path $reportWide `
        --log-path $logWide `
        --no-lock
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: d=256 seed=$seed failed with exit code $LASTEXITCODE"
        continue
    }
    
    $r = Get-Content $reportWide | ConvertFrom-Json
    $results += [PSCustomObject]@{
        Config = "recurrent_4_d256"
        Seed = $seed
        ValLoss = $r.conditions.recurrent_4.val_loss
        Params = $r.conditions.recurrent_4.params
        Time = $r.conditions.recurrent_4.train_time_seconds
    }
    $results += [PSCustomObject]@{
        Config = "distinct_4_d256"
        Seed = $seed
        ValLoss = $r.conditions.distinct_4.val_loss
        Params = $r.conditions.distinct_4.params
        Time = $r.conditions.distinct_4.train_time_seconds
    }
    Write-Host "  recurrent_4_d256: val=$($r.conditions.recurrent_4.val_loss)"
    Write-Host "  distinct_4_d256:  val=$($r.conditions.distinct_4.val_loss)"
    Write-Host ""

    # --- d=128 run (primary: distinct_4_d128) ---
    $reportNarrow = "$artifactDir\verify_narrow_seed${seed}.json"
    $logNarrow = "runs\verify_narrow_seed${seed}.log"
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting d=128, seed=$seed..."
    
    & $python -m runs.recurrent_depth_lm `
        --seed $seed `
        --d-model 128 --n-heads 4 --ff-dim 512 `
        --train-characters 900000 --val-characters 100000 `
        --report-path $reportNarrow `
        --log-path $logNarrow `
        --no-lock
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: d=128 seed=$seed failed with exit code $LASTEXITCODE"
        continue
    }
    
    $r = Get-Content $reportNarrow | ConvertFrom-Json
    $results += [PSCustomObject]@{
        Config = "distinct_4_d128"
        Seed = $seed
        ValLoss = $r.conditions.distinct_4.val_loss
        Params = $r.conditions.distinct_4.params
        Time = $r.conditions.distinct_4.train_time_seconds
    }
    $results += [PSCustomObject]@{
        Config = "recurrent_4_d128"
        Seed = $seed
        ValLoss = $r.conditions.recurrent_4.val_loss
        Params = $r.conditions.recurrent_4.params
        Time = $r.conditions.recurrent_4.train_time_seconds
    }
    Write-Host "  distinct_4_d128:  val=$($r.conditions.distinct_4.val_loss)"
    Write-Host "  recurrent_4_d128: val=$($r.conditions.recurrent_4.val_loss)"
    Write-Host ""
}

$elapsed = (Get-Date) - $startTime
Write-Host "=== All experiments complete in $([math]::Round($elapsed.TotalMinutes, 1)) minutes ==="
Write-Host ""
Write-Host "=== SUMMARY ==="

# Group and summarize
$grouped = $results | Group-Object Config
foreach ($g in $grouped) {
    $vals = $g.Group | ForEach-Object { $_.ValLoss }
    $mean = ($vals | Measure-Object -Average).Average
    $count = $vals.Count
    if ($count -gt 1) {
        $variance = ($vals | ForEach-Object { ($_ - $mean) * ($_ - $mean) } | Measure-Object -Sum).Sum / ($count - 1)
        $std = [math]::Sqrt($variance)
    } else {
        $std = 0
    }
    Write-Host ("{0,-20} mean={1:F6} std={2:F6} n={3} vals=[{4}]" -f $g.Name, $mean, $std, $count, (($vals | ForEach-Object { "{0:F6}" -f $_ }) -join ", "))
}

Write-Host ""
Write-Host "KEY COMPARISON: recurrent_4_d256 vs distinct_4_d128"
$recVals = ($results | Where-Object { $_.Config -eq "recurrent_4_d256" }).ValLoss
$distVals = ($results | Where-Object { $_.Config -eq "distinct_4_d128" }).ValLoss
if ($recVals -and $distVals) {
    $recMean = ($recVals | Measure-Object -Average).Average
    $distMean = ($distVals | Measure-Object -Average).Average
    $delta = $recMean - $distMean
    Write-Host ("  recurrent_4_d256 mean: {0:F6}" -f $recMean)
    Write-Host ("  distinct_4_d128  mean: {0:F6}" -f $distMean)
    Write-Host ("  Delta (rec - dist):    {0:F6}" -f $delta)
    if ($delta -lt 0) {
        Write-Host "  RECURRENT WINS (negative delta = lower loss)"
    } else {
        Write-Host "  DISTINCT WINS (positive delta = recurrent has higher loss)"
    }
}

# Save summary to file
$summaryPath = "$artifactDir\multiseed_summary.json"
$results | ConvertTo-Json -Depth 3 | Set-Content $summaryPath
Write-Host ""
Write-Host "Summary saved to: $summaryPath"
