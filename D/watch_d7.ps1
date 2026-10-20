# Progress viewer for D7 (Nasdaq-100 back to 1995). Read-only. Ctrl+C to stop.
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File B:\qfin-main\D\watch_d7.ps1
$root = $PSScriptRoot
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
function Count($dir) { if (Test-Path $dir) { @(Get-ChildItem $dir -Recurse -Filter '*.json' -File -ErrorAction SilentlyContinue).Count } else { 0 } }
function Bar($n, $total) { $pct = if ($total -gt 0) { [math]::Min(100, [math]::Round(100 * $n / $total, 1)) } else { 0 }; ('#' * [int]($pct / 2.5)).PadRight(40, '.') + ("  {0,5}%  {1}/{2}" -f $pct, $n, $total) }
function Lines($file) { if (Test-Path $file) { @(Get-Content $file -Encoding UTF8 -ErrorAction SilentlyContinue | Where-Object { $_ -notmatch 'Warning' }) } else { @() } }
while ($true) {
    $o = New-Object System.Collections.Generic.List[string]
    $o.Add('D7: quantum strategies on the Nasdaq-100 back to 1995   ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '   (refresh 15 s, Ctrl+C to stop)')
    $o.Add('')
    $vol = Count "$root\results\d7\vol"; $e2e = Count "$root\results\d7\e2e"
    $o.Add('  volatility forecasters (QLSTM/QFWP + classical) : ' + (Bar $vol 3200))
    $o.Add('  end-to-end position sizing (QLSTM + classical)   : ' + (Bar $e2e 2560))
    $gpuLines = @(); foreach ($f in 'd7_gpu2.log', 'd7_gpu3.log', 'd7_gpu4.log', 'd7_gpu5.log', 'd7_gpu6.log', 'd7_gpu7.log') { $gpuLines += @(Lines "$root\logs\$f") }
    $all = @(Lines "$root\logs\d7.log") + @(Lines "$root\logs\d7_cpu2.log") + $gpuLines
    $g = @($all | Where-Object { $_ -match 'models' })
    $o.Add("  groups finished : $($g.Count) / 576   (on GPU: $(@($gpuLines | Where-Object { $_ -match 'models' }).Count))")
    $o.Add('  latest:')
    $g | Select-Object -Last 6 | ForEach-Object { $o.Add('    ' + $_) }
    $err = @($all | Where-Object { $_ -match 'Traceback|failed|Error' })
    if ($err.Count) { $o.Add('  !! errors: ' + $err[-1]) }
    if (@(Lines "$root\logs\d7_cpu2.log" | Where-Object { $_ -match 'D7 done' }).Count) { $o.Add('  CPU workers DONE') }
    $procs = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'd7_ndx|multiprocessing' })
    $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    $o.Add('')
    $o.Add("== python processes: $($procs.Count)    CPU load: $cpu %")
    Clear-Host
    $o | ForEach-Object { Write-Host $_ }
    Start-Sleep -Seconds 15
}
