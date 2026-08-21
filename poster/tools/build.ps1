# Build the A0 poster and its A3 reduction.
#
# Not latexmk: MiKTeX ships latexmk as a Perl script wrapper and Perl is not
# installed on this machine, so it fails immediately.
#
# The three pdflatex flags are all load-bearing. -interaction=nonstopmode stops
# TeX asking about errors, -halt-on-error stops it grinding through hundreds of
# cascading ones, and -disable-installer stops MiKTeX raising a modal dialog on
# the first missing file — [MPM]AutoInstall is 2 (Undecided) here, and
# [MPM]RemoteRepository is empty, so a script would simply hang forever.

$ErrorActionPreference = 'Stop'

$MIK   = 'C:\Users\Jorge\AppData\Local\Programs\MiKTeX\miktex\bin\x64'
$ROOT  = Split-Path -Parent $PSScriptRoot
$TEX   = Join-Path $MIK 'pdflatex.exe'
$FLAGS = @('-interaction=nonstopmode','-halt-on-error','-file-line-error',
           '-disable-installer')

Set-Location $ROOT
New-Item -ItemType Directory -Force -Path 'build','tikz\build' | Out-Null

# $src, not $tex: PowerShell variables are case-insensitive, so a parameter
# named $tex silently shadows the script-level $TEX holding pdflatex's path,
# and the script ends up trying to execute the .tex file as a document.
function Invoke-Tex($src, $outdir, $label) {
    & $TEX @FLAGS "-output-directory=$outdir" $src | Out-Null
    # Never gate on $? here: every MiKTeX binary writes "unsupported version of
    # Windows" to stderr on every call, and PowerShell 5.1 turns redirected
    # native stderr into a NativeCommandError with $? = $false even on exit 0.
    if ($LASTEXITCODE -ne 0) {
        $log = Join-Path $outdir ([IO.Path]::GetFileNameWithoutExtension($src) + '.log')
        Write-Host "FAILED: $label" -ForegroundColor Red
        if (Test-Path $log) {
            Select-String -Path $log -Pattern '^.*:\d+:.*$' |
                Select-Object -First 12 | ForEach-Object { Write-Host "  $_" }
        }
        throw "$label failed (see $log)"
    }
}

Write-Host '--- TikZ diagrams ---'
Get-ChildItem "$ROOT\tikz\*.tex" | Where-Object { $_.Name -notlike '_*' } |
  ForEach-Object {
    Write-Host "  $($_.Name)"
    Invoke-Tex "tikz/$($_.Name)" 'tikz\build' "tikz/$($_.Name)"
  }

Write-Host '--- Poster (A0), two passes for the full-bleed header band ---'
1..2 | ForEach-Object {
    Write-Host "  pass $_"
    Invoke-Tex 'poster.tex' 'build' 'poster.tex'
}

Write-Host '--- A3 reduction ---'
Copy-Item 'build\poster.pdf' 'poster.pdf' -Force
Invoke-Tex 'poster-a3.tex' 'build' 'poster-a3.tex'
Remove-Item 'poster.pdf' -Force

foreach ($f in 'build\poster.pdf','build\poster-a3.pdf') {
    if (-not (Test-Path $f)) { throw "missing output: $f" }
    $kb = [math]::Round((Get-Item $f).Length / 1KB)
    Write-Host ("OK  {0}  {1} KB" -f $f, $kb) -ForegroundColor Green
}
