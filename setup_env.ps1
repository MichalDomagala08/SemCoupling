# Recreates the .venv for SemCoupling from scratch.
# Usage (from the SemCoupling folder):
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1          # core only
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Nlp     # core + NLP stack

param([switch]$Nlp)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Get-Command deactivate -ErrorAction SilentlyContinue) { deactivate }

if (Test-Path .venv) {
    Write-Host "Removing old .venv ..."
    Remove-Item -Recurse -Force .venv
}

Write-Host "Creating .venv with Python 3.11 ..."
py -3.11 -m venv .venv
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
if ($Nlp) { & $py -m pip install -r requirements-nlp.txt }

# Register the kernel so VS Code / Jupyter shows it by name
& $py -m ipykernel install --user --name semcoupling --display-name "Python (SemCoupling)"

Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host "In VS Code pick kernel: 'Python (SemCoupling)'"
