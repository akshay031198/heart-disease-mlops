# End-to-end local pipeline (Windows PowerShell). Run from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
$ErrorActionPreference = "Stop"
function Run($cmd) {
    Write-Host ">> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $cmd" }
}
Run "python -m heart_disease.download_data"
Run "python -m heart_disease.eda"
Run "flake8 heart_disease api tests scripts"
Run "pytest"
Run "python -m heart_disease.train"
Run "python -m heart_disease.predict --input samples/sample_request.json"
Write-Host "Pipeline finished. Start the API with: uvicorn api.main:app --host 0.0.0.0 --port 8000" -ForegroundColor Green
