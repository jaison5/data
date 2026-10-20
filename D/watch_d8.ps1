# Progress viewer for D7 (Nasdaq-100 back to 1995). Read-only. Ctrl+C to stop.
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File B:\qfin-main\D\watch_d7.ps1
$root = $PSScriptRoot
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
function Count($dir) { if (Test-Path $dir) { @(Get-ChildItem $dir -Recurse -Filter '*.json' -File -ErrorAction SilentlyContinue).Count } else { 0 } }
function Bar($n, $total) { $pct = if ($total -gt 0) { [math]::Min(100, [math]::Round(100 * $n / $total, 1)) } else { 0 }; ('#' * [int]($pct / 2.5)).PadRight(40, '.') + ("  {0,5}%  {1}/{2}" -f $pct, $n, $total) }
function Lines($file) { if (Test-Path $file) { @(Get-Content $file -Encoding UTF8 -ErrorAction SilentlyContinue | Where-Object { $_ -notmatch 'Warning' }) } else { @() } }
while ($true) {
    $o = New-Object System.Collections.Generic.List[string]
    $o.Add('D8: is the quantum structure necessary? (regularized classical, Fourier features, learning curves)   ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '   (refresh 15 s, Ctrl+C to stop)')
    $o.Add('')
    $reg = 0; foreach ($t in "reg_early","reg_wd","reg_turn") { $reg += Count "$root\results\d8\$t" }; $ff = Count "$root\results\d8\ff_std"; $lc = 0; foreach ($t in "lc_K2","lc_K4","lc_K8") { $lc += Count "$root\results\d8\$t" }
    $o.Add('  (2) regularization-matched classical : ' + (Bar $reg 2850))
    $o.Add('  (3) Fourier-feature classical LSTM   : ' + (Bar $ff 760))
    $o.Add('  (4) learning curves K=2/4/8 years     : ' + (Bar $lc 4560))
    $gpuLines = @(); foreach ($f in 'd8_gpu1.log', 'd8_gpu2.log', 'd8_cpu_q.log') { $gpuLines += @(Lines "$root\logs\$f") }
    $all = @(Lines "$root\logs\d8_cpu.log") + $gpuLines
    $g = @($all | Where-Object { $_ -match 'models' })
    $o.Add("  groups finished : $($g.Count) / 817   (on GPU: $(@($gpuLines | Where-Object { $_ -match 'models' }).Count))")
    $o.Add('  latest:')
    $g | Select-Object -Last 6 | ForEach-Object { $o.Add('    ' + $_) }
    $err = @($all | Where-Object { $_ -match 'Traceback|failed|Error' })
    if ($err.Count) { $o.Add('  !! errors: ' + $err[-1]) }
    if (@(Lines "$root\logs\d8_cpu.log" | Where-Object { $_ -match 'D8 done' }).Count) { $o.Add('  CPU workers DONE') }
    $procs = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'd8_controls|multiprocessing' })
    $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    $o.Add('')
    $o.Add("== python processes: $($procs.Count)    CPU load: $cpu %")
    Clear-Host
    $o | ForEach-Object { Write-Host $_ }
    Start-Sleep -Seconds 15
}
