# Verification gate for the built poster. Every tool used here already ships
# with MiKTeX (poppler 23.13.0), so there is nothing to install.

$MIK  = 'C:\Users\Jorge\AppData\Local\Programs\MiKTeX\miktex\bin\x64'
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$fail = 0

function Say($ok, $msg) {
    if ($ok) { Write-Host "  PASS  $msg" -ForegroundColor Green }
    else     { Write-Host "  FAIL  $msg" -ForegroundColor Red; $script:fail++ }
}

Write-Host 'Page geometry'
$a0 = & "$MIK\pdfinfo.exe" 'build\poster.pdf'
$a3 = & "$MIK\pdfinfo.exe" 'build\poster-a3.pdf'
Say (($a0 -join ' ') -match '2383\.9\d+ x 3370\.3\d+') 'A0 is 841 x 1189 mm'
Say (($a0 -join ' ') -match 'Pages:\s+1\b')            'A0 is one page'
Say (($a3 -join ' ') -match '841\.\d+ x 1190\.\d+')    'A3 is 297 x 420 mm'

# A minipage taller than the frame does NOT raise an Overfull \vbox in beamer
# and does not spill onto a second page — it simply runs off the bottom of the
# sheet. These measurements are the only thing that catches it.
Write-Host 'Column fit'
$seen = @{}
Select-String -Path 'build\poster.log' -Pattern 'COLHEIGHT: column (\d) = ([\d.]+)pt\s+of ([\d.]+)pt' |
  ForEach-Object {
    $c  = $_.Matches[0].Groups[1].Value
    if ($seen.ContainsKey($c)) { return }
    $seen[$c] = $true
    $h  = [double]$_.Matches[0].Groups[2].Value
    $mx = [double]$_.Matches[0].Groups[3].Value
    $pc = 100 * $h / $mx
    Say ($h -le $mx) ("column {0} fills {1:N1} % of the sheet" -f $c, $pc)
  }

Write-Host 'Figure resolution (run on the A0; the A3 is 2.83x finer and meaningless here)'
$rows = & "$MIK\pdfimages.exe" -list 'build\poster.pdf' | Select-Object -Skip 2
$low = @($rows | Where-Object {
    $f = $_ -split '\s+' | Where-Object { $_ }
    $f[2] -eq 'image' -and ([int]$f[12]) -lt 150 })
Say ($low.Count -eq 0) "every raster is at least 150 dpi at print size"
if ($low.Count) { $low | ForEach-Object { Write-Host "        $_" } }

Write-Host 'Fonts'
$fonts = & "$MIK\pdffonts.exe" 'build\poster.pdf' | Select-Object -Skip 2
# Match the emb/sub/uni triple by position from the right, not by field index:
# a Type 3 row splits into a different number of fields than a Type 1 row, so
# a fixed index silently reads the `sub' column and reports a false failure.
$notEmb = @($fonts | Where-Object { $_ -match '\s(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$' -and
                                    $Matches[1] -eq 'no' })
$type3  = @($fonts | Where-Object { $_ -match 'Type 3' })
Say ($notEmb.Count -eq 0) 'all fonts embedded'
# Type 3 here means a METAFONT bitmap: jagged at a metre, and print shops
# reject them.
Say ($type3.Count -eq 0) 'no Type 3 bitmap fonts'
if ($type3.Count) { $type3 | ForEach-Object { Write-Host "        $_" } }

Write-Host 'Overflow'
$over = @(Select-String -Path 'build\poster.log' -Pattern 'Overfull \\hbox')
Say ($over.Count -eq 0) 'no overfull hboxes'

Write-Host ''
if ($fail) { Write-Host "$fail check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host 'All checks passed' -ForegroundColor Green
