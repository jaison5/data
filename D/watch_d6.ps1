# Progress viewer for D6 (end-to-end Sharpe-trained QLSTM / QFWP / classical). Read-only. Ctrl+C to stop.
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File B:\qfin-main\D\watch_d6.ps1
$root = $PSScriptRoot
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
function Count($dir, $filter) { if (Test-Path $dir) { @(Get-ChildItem $dir -Recurse -Filter $filter -File -ErrorAction SilentlyContinue).Count } else { 0 } }
function Bar($n, $total) { $pct = if ($total -gt 0) { [math]::Min(100, [math]::Round(100 * $n / $total, 1)) } else { 0 }; ('#' * [int]($pct / 2.5)).PadRight(40, '.') + ("  {0,5}%  {1}/{2}" -f $pct, $n, $total) }
function Tail($file, $n) { if (Test-Path $file) { @(Get-Content $file -Encoding UTF8 -ErrorAction SilentlyContinue | Select-Object -Last $n) } else { @() } }
$log = "$root\logs\d6.log"
while ($true) {
    $o = New-Object System.Collections.Generic.List[string]
    $o.Add('D6 end-to-end training   ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '   (refresh every 15 s, Ctrl+C to stop)')
    if (Test-Path $log) { $o.Add('  started ' + (Get-Item $log).CreationTime.ToString('HH:mm') + ', running ' + [int]((Get-Date) - (Get-Item $log).CreationTime).TotalMinutes + ' min') }
    $o.Add('')
    $all = Count "$root\results\d6" '*.json'
    $o.Add('  all trainings : ' + (Bar $all 4560))
    $o.Add('  per model (2 caps x 19 test years x 20 runs = 760 each):')
    foreach ($m in 'QLSTM', 'QFWP', 'FWPC', 'LSTM3', 'LSTM1', 'LINEAR') {
        $o.Add(('    {0,-7}' -f $m) + (Bar (Count "$root\results\d6" "$($m)_*.json") 760))
    }
    $locks = if (Test-Path "$root\results\_locks_d6") { @(Get-ChildItem "$root\results\_locks_d6").Count } else { 0 }
    $done = @(Tail $log 1000 | Where-Object { $_ -match ' models, ' })
    $o.Add("  configs started : $locks / 228   (QLSTM first: 38 configs, ~25-40 min per group of 15; then QFWP and the classical models, fast)")
    $o.Add("  groups finished : $($done.Count) / 456   (each config = 15 models for the learning-rate choice + 5 more seeds)")
    $o.Add('  latest:')
    $done | Select-Object -Last 8 | ForEach-Object { $o.Add('    ' + $_) }
    $err = @(Tail $log 1000 | Where-Object { $_ -match 'Traceback|failed|Error' })
    if ($err.Count) { $o.Add('  !! errors: ' + $err[-1]) }
    if (@(Tail $log 5 | Where-Object { $_ -match 'D6 done' }).Count) { $o.Add('  D6 TRAINING DONE (report: python d6_e2e.py --report)') }
    $o.Add('')
    $procs = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'd6_e2e|multiprocessing' })
    $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    $o.Add("== python training processes: $($procs.Count)    CPU load: $cpu %")
    Clear-Host
    $o | ForEach-Object { Write-Host $_ }
    Start-Sleep -Seconds 15
}
