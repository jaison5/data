# Progress viewer for D9 (noise and finite shots). Read-only. Ctrl+C to stop.
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File B:\qfin-main\D\watch_d9.ps1
$root = $PSScriptRoot
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
function Count($dir, $pat) { if (Test-Path $dir) { @(Get-ChildItem $dir -Filter $pat -File -ErrorAction SilentlyContinue).Count } else { 0 } }
function Bar($n, $total) { $pct = if ($total -gt 0) { [math]::Min(100, [math]::Round(100 * $n / $total, 1)) } else { 0 }; ('#' * [int]($pct / 2.5)).PadRight(40, '.') + ("  {0,5}%  {1}/{2}" -f $pct, $n, $total) }
while ($true) {
    $o = New-Object System.Collections.Generic.List[string]
    $o.Add('D9: quantum models under hardware noise and finite shots   ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '   (refresh 10 s, Ctrl+C to stop)')
    $o.Add('')
    $o.Add('  retrain + save parameters')
    foreach ($j in 'vol_QLSTM', 'vol_QFWP', 'e2e_QLSTM') { $o.Add(('    {0,-10}: ' -f $j) + (Bar (Count "$root\results\d9\params" "$($j)_y*.pt") 19)) }
    $o.Add('  noisy inference (80 settings x realizations per model)')
    foreach ($j in 'vol_QLSTM', 'vol_QFWP', 'e2e_QLSTM') { $o.Add(('    {0,-10}: ' -f $j) + (Bar (Count "$root\results\d9\pred" "$($j)_*.npy") 80)) }
    $o.Add('  latest:')
    Get-ChildItem "$root\logs" -Filter 'd9_*.log' -ErrorAction SilentlyContinue | ForEach-Object {
        $l = @(Get-Content $_.FullName -Encoding UTF8 -ErrorAction SilentlyContinue | Where-Object { $_ -notmatch 'Warning|self.P' }) | Select-Object -Last 2
        $l | ForEach-Object { $o.Add('    ' + $_) }
    }
    if (Test-Path "$root\results\d9_verdict.csv") { $o.Add(''); $o.Add('  REPORT DONE: results\d9_verdict.csv') }
    Clear-Host
    $o | ForEach-Object { Write-Host $_ }
    Start-Sleep -Seconds 10
}
