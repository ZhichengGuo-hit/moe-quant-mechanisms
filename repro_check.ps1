param(
    [string]$Model = "",
    [string]$RepoRoot = "",
    [switch]$Build
)
# Read-only self-check: interpreter/deps/patch/result table; optional build and smoke.
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\repro_check.ps1
#   powershell -ExecutionPolicy Bypass -File .\repro_check.ps1 -Model E:\path\model.gguf
#   powershell -ExecutionPolicy Bypass -File .\repro_check.ps1 -Build

$ErrorActionPreference = "Continue"
$here = $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = Split-Path $here -Parent }

function Section($t) { Write-Host ""; Write-Host ("== " + $t + " ==") -ForegroundColor Cyan }

Section "Python / CUDA / CMake"
python -V
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cmake --version | Select-Object -First 1

Section "Python deps (L1)"
python -c "import numpy,pandas,pyarrow;print('core deps ok')"

Section "moe_lib (needs torch)"
python -c "import sys;sys.path.insert(0, r'$RepoRoot\tcq_exp');import moe_lib;print('moe_lib ok')"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "moe_lib import failed: torch missing in this interpreter (see REPRODUCIBILITY.md sections 8/9)."
}

Section "llama.cpp patches"
foreach ($p in @("llama.cpp-tcmoe.patch", "llama-upstream.patch")) {
    $f = Join-Path $here ("llamacpp\" + $p)
    if (Test-Path $f) {
        $kb = [math]::Round((Get-Item $f).Length / 1KB, 1)
        Write-Host ("OK   {0} ({1} KB)" -f $p, $kb)
    } else {
        Write-Warning ("missing " + $p)
    }
}
$repo = Join-Path $RepoRoot "llama.cpp"
if (Test-Path (Join-Path $repo ".git")) {
    git -C $repo apply --check --reverse (Join-Path $here "llamacpp\llama.cpp-tcmoe.patch") 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "llama.cpp: patch applied (reverse check passed)" }
    else { Write-Host "llama.cpp: patch not applied or tree modified (skipped)" }
}

Section "Result table"
$csv = Join-Path $here "DATA_TABLE.csv"
if (Test-Path $csv) {
    python -c "import csv;rows=list(csv.DictReader(open(r'$csv',encoding='utf-8-sig')));print('DATA_TABLE rows=%d' % len(rows))"
} else {
    Write-Warning "DATA_TABLE.csv not found"
}

if ($Build) {
    Section "Build llama.cpp (CUDA)"
    $msvc = Join-Path $RepoRoot "coserving\msvc_env.ps1"
    if (Test-Path $msvc) { . $msvc }
    $src = Join-Path $RepoRoot "llama.cpp"
    $bld = Join-Path $src "build-cuda"
    cmake -S $src -B $bld -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -eq 0) {
        cmake --build $bld --config Release -j 16 --target llama-server llama-quantize llama-route-dump
    }
}

if ($Model) {
    Section "Smoke test"
    $cli = Get-ChildItem (Join-Path $RepoRoot "llama.cpp") -Recurse -Filter "llama-cli.exe" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (($null -ne $cli) -and (Test-Path $Model)) {
        & $cli.FullName -m $Model -p "1+1=" -n 8 -ngl 99 --no-warmup
    } else {
        Write-Warning ("llama-cli.exe or model not found: " + $Model)
    }
}

Write-Host ""
Write-Host "Self-check finished." -ForegroundColor Green
