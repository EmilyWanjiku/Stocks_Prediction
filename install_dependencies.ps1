param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not available on PATH. Please install Python 3.10+ and try again."
    exit 1
}

$pythonExe = (Get-Command python).Source

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment at: $VenvPath"
    & $pythonExe -m venv $VenvPath
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment python executable not found at $venvPython"
    exit 1
}

Write-Host "Using virtual environment Python at: $venvPython"

$packages = @(
    "numpy",
    "pandas",
    "scikit-learn",
    "tensorflow",
    "statsmodels",
    "xgboost",
    "joblib",
    "matplotlib",
    "plotly",
    "streamlit",
    "openpyxl",
    "werkzeug"
)

Write-Host "`nChecking pip availability..."
& $venvPython -m pip --version | Out-Null

Write-Host "Upgrading pip (user scope)..."
& $venvPython -m pip install --upgrade pip

foreach ($package in $packages) {
    Write-Host "`nInstalling $package..."
    & $venvPython -m pip install $package
}

Write-Host "`nAll packages processed."
Write-Host "Virtual environment ready."
Write-Host "Activate with: `n`t.\$VenvPath\Scripts\Activate.ps1"
Write-Host "Then train models with: `n`tpython train_models.py"
Write-Host "Launch the dashboard with: `n`tstreamlit run streamlit_app.py"

