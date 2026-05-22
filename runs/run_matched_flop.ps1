Set-Location $PSScriptRoot\..
$env:PYTHONUNBUFFERED = "1"
& .\.venv\Scripts\python.exe -u -m experiments.fixed_multi_rate.matched_flop --device cuda --seed 42 --output-dir experiments/fixed_multi_rate/artifacts/matched_flop *> runs\matched_flop_output.log
