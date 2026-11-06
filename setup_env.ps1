# Recreates the .venv for SemCoupling from scratch (Windows).
# Usage (from the SemCoupling folder):
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1          # core only
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Nlp     # core + NLP stack (~2-3 GB)
# macOS / Linux: use setup_env.sh instead.

param([switch]$Nlp)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Get-Command deactivate -ErrorAction SilentlyContinue) { deactivate }

if (Test-Path .venv) {
    Write-Host "Removing old .venv ..."
    Remove-Item -Recurse -Force .venv
}

# Pick a Python >= 3.11 (pandas 3 needs it) via the 'py' launcher, falling back to 'python'
$created = $false
foreach ($v in @("-3.11", "-3.12", "-3.13")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py $v -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Creating .venv with Python $v ..."
            & py $v -m venv .venv
            $created = $true; break
        }
    }
}
if (-not $created) {
    Write-Host "Creating .venv with 'python' from PATH ..."
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
    if ($LASTEXITCODE -ne 0) { throw "Python >= 3.11 not found. Install it from python.org." }
    python -m venv .venv
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
if ($Nlp) {
    & $py -m pip install -r requirements-nlp.txt
    & $py -m nltk.downloader punkt punkt_tab      # Polish tokenizer models for NLTK
}

# Register the kernel so VS Code / Jupyter shows it by name
& $py -m ipykernel install --user --name semcoupling --display-name "Python (SemCoupling)"

Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host "In VS Code pick kernel: 'Python (SemCoupling)'"
