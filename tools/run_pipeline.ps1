param(
    [string]$ConfigPath = "config/project.json"
)

$activatePath = ".\\.venv\\Scripts\\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    Write-Error "No se encontro .venv. Ejecuta: python -m venv .venv"
    exit 1
}

& $activatePath
python -m src.pipeline $ConfigPath
