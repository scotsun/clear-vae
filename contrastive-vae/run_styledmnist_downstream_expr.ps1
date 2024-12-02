# run styled-mnist expr

for ($i = 0; $i -lt 11; $i++) {
    Write-Host "Running expr $i`n"
    python run_styledmnist_downstream_expr.py
    Write-Host "Complete expr $i`n"
}