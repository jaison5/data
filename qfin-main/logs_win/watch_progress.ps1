# Progress viewer for the Windows rerun. Read-only: it only reads logs/results. Ctrl+C to stop.
$root = Split-Path -Parent $PSScriptRoot   # repo folder (this script lives in logs_win\)
$log = Join-Path $root 'logs_win'
while ($true) {
    $out = New-Object System.Collections.Generic.List[string]
    $out.Add('qfin rerun progress   ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '   (refresh every 30 s, Ctrl+C to stop)')
    $out.Add('')
    $out.Add('== Exp 2 pipeline (forecaster -> A3C -> QA3C -> report)')
    $drv = Join-Path $log 'exp2_driver.log'
    if (Test-Path $drv) { Get-Content $drv -Encoding UTF8 | ForEach-Object { $out.Add('  ' + $_) } }
    $out.Add('')
    $out.Add('== QA3C training (3 seeds run together, 8 workers each, 1600 episodes per seed)')
    foreach ($s in 0, 1, 2) {
        $f = Join-Path $log "pl_QA3C_s$s.log"
        if (-not (Test-Path $f)) { $out.Add("  seed $s : not started"); continue }
        $lines = Get-Content $f -Encoding UTF8
        $err = $lines | Where-Object { $_ -match 'Traceback|Error' } | Select-Object -First 1
        $last = $lines | Where-Object { $_ -match '/1600' } | Select-Object -Last 1
        $done = $lines | Where-Object { $_ -match '1600 episodes' -and $_ -notmatch '/1600' } | Select-Object -Last 1
        if ($err) { $out.Add("  seed $s : ERROR -> $err"); continue }
        if ($last -match '(\d+)/1600') {
            $n = [int]$Matches[1]
            $pct = [math]::Round(100 * $n / 1600, 1)
            $bar = ('#' * [int]($pct / 2.5)).PadRight(40, '.')
            $eta = ''
            if ($last -match '([\d.]+)\s*\S+$' -and $n -gt 0) {
                $min = [double]$Matches[1]
                $left = $min * (1600 - $n) / $n
                $eta = '  elapsed {0:N0} min, ETA ~{1:N0} min ({2})' -f $min, $left, (Get-Date).AddMinutes($left).ToString('HH:mm')
            }
            $out.Add(("  seed {0} : [{1}] {2,5}%  {3}/1600{4}" -f $s, $bar, $pct, $n, $eta))
            $out.Add('           ' + $last.Trim())
        }
        $test = $lines | Where-Object { $_ -match 'QA3C s\d .*%.*%' -and $_ -notmatch '/1600' } | Select-Object -Last 1
        if ($test) { $out.Add('           TEST: ' + $test.Trim()) }
    }
    $tdir = Join-Path $root 'results_win\exp1_timing_vjptrue'
    if (Test-Path $tdir) {
        $k = @(Get-ChildItem $tdir -Filter '*_s*.json').Count
        $out.Add('')
        $out.Add("== Supplementary timing (device_vjp=True): $k / 36 runs done")
    }
    $out.Add('')
    $procs = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'pl_exp|multiprocessing' })
    $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    $out.Add("== Experiment python processes: $($procs.Count)    CPU load: $cpu %")
    Clear-Host
    $out | ForEach-Object { Write-Host $_ }
    Start-Sleep -Seconds 30
}
