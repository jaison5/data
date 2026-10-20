# Progress viewer for direction D (D1 direction models, D4 quantum volatility models). Read-only. Ctrl+C to stop.
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File B:\qfin-main\D\watch_d.ps1
$root = $PSScriptRoot
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
function Count($dir) { if (Test-Path $dir) { @(Get-ChildItem $dir -Recurse -Filter '*.json' -File -ErrorAction SilentlyContinue).Count } else { 0 } }
function Bar($n, $total) { $pct = if ($total -gt 0) { [math]::Min(100, [math]::Round(100 * $n / $total, 1)) } else { 0 }; ('#' * [int]($pct / 2.5)).PadRight(40, '.') + ("  {0,5}%  {1}/{2}" -f $pct, $n, $total) }
function Tail($file, $n) { if (Test-Path $file) { @(Get-Content $file -Encoding UTF8 -ErrorAction SilentlyContinue | Where-Object { $_ -notmatch 'Warning' } | Select-Object -Last $n) } else { @() } }
$t0 = Get-Date
while ($true) {
    $o = New-Object System.Collections.Generic.List[string]
    $o.Add('Direction D training   ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '   (refresh every 15 s, Ctrl+C to stop)')
    $o.Add('')
    $d1 = Count "$root\results\d1"
    $o.Add('== D1  direction models on SPY/QQQ (quantum role 2)')
    $o.Add('  trainings : ' + (Bar $d1 3838) + $(if ($d1 -ge 3838) { '   done, report written' } else { '' }))
    $o.Add('')
    $d4 = Count "$root\results\d4"
    $locks = if (Test-Path "$root\results\_locks_d4") { @(Get-ChildItem "$root\results\_locks_d4").Count } else { 0 }
    $o.Add('== D4  volatility forecasting with QLSTM / QFWP / classical (quantum role 1)')
    $o.Add('  trainings : ' + (Bar $d4 3800))
    $o.Add("  configs started : $locks / 190   (QLSTM first: 38 configs x 2 stages, then QFWP / FWPC / LSTM)")
    $done = @(@(Tail "$root\logs\d4.log" 400) + @(Tail "$root\logs\d4_gpu.log" 400) | Where-Object { $_ -match 'models' })
    $gpu = @(Tail "$root\logs\d4_gpu.log" 400 | Where-Object { $_ -match 'models' }).Count
    $o.Add("  of which on GPU : $gpu groups")
    $o.Add("  groups finished : $($done.Count) / 380")
    $o.Add('  latest:')
    $done | Select-Object -Last 6 | ForEach-Object { $o.Add('    ' + $_) }
    $err = @(Tail "$root\logs\d4.log" 400 | Where-Object { $_ -match 'Traceback|failed|Error' })
    if ($err.Count) { $o.Add('  !! errors: ' + $err[-1]) }
    if (@(Tail "$root\logs\d4.log" 5 | Where-Object { $_ -match 'D4 done' }).Count) { $o.Add('  D4 DONE') }
    $o.Add('')
    $procs = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'd4_vol|run_d1|multiprocessing' })
    $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    $o.Add("== python training processes: $($procs.Count)    CPU load: $cpu %")
    Clear-Host
    $o | ForEach-Object { Write-Host $_ }
    Start-Sleep -Seconds 15
}
