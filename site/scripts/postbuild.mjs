// Post-build steps for a static host (GitHub Pages), run by `npm run build` after `vite build`:
//
//   1. .nojekyll, so GitHub Pages serves the files as they are;
//   2. licenses/: the licence texts of the packages bundled into the site, one file each plus one combined file;
//   3. a scan of the text files for local folders, file URLs and IP addresses (the build fails on a hit);
//   4. a size report; the build fails above 30 MB in all or 5 MB for one file.
//
// The site is one page (index.html) whose state lives in the URL hash, so no other address needs a page of its own
// and GitHub Pages needs no 404 fallback.
//
// The output folder is SITE_OUT_DIR (default ../docs), as in vite.config.js (see scripts/site-env.mjs).
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { extname, join, relative, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { siteEnv } from './site-env.mjs'

const siteDir = fileURLToPath(new URL('..', import.meta.url))
const { outDir } = siteEnv(siteDir)
const MAX_TOTAL = 30e6 // bytes, the whole site
const MAX_FILE = 5e6 // bytes, one file

const rel = (p) => relative(outDir, p).split(sep).join('/')
const fail = (msg) => {
  console.error(`postbuild: ${msg}`)
  process.exit(1)
}

if (!existsSync(join(outDir, 'index.html'))) fail(`no index.html in the output folder; did vite build run?`)

// ---------------------------------------------------------------------------------------------------- 1: .nojekyll
const written = []
writeFileSync(join(outDir, '.nojekyll'), '')
written.push('.nojekyll')

// ---------------------------------------------------------------------------------------------------- 2: licences
// Packages that can end up in the bundle: every runtime (non-dev) package of package-lock.json, plus the build
// tools whose code is inlined (Vite's preload helper, Rolldown's module helpers, Tailwind's base styles).
const lock = JSON.parse(readFileSync(join(siteDir, 'package-lock.json'), 'utf8'))
const BUILD_HELPERS = ['vite', 'rolldown', 'tailwindcss']
const packages = new Map() // name -> { version, license, dir }
for (const [path, info] of Object.entries(lock.packages ?? {})) {
  if (!path.startsWith('node_modules/')) continue
  const name = path.slice(path.lastIndexOf('node_modules/') + 'node_modules/'.length)
  const runtime = !info.dev && !info.devOptional
  if (!runtime && !BUILD_HELPERS.includes(name)) continue
  if (!packages.has(name)) packages.set(name, { version: info.version, license: info.license, dir: join(siteDir, path) })
}

const LICENCE_FILE = /^(licen[sc]e|copying|notice)([-_.][a-z0-9]+)*(\.(md|txt|markdown))?$/i
const licDir = join(outDir, 'licenses')
mkdirSync(licDir, { recursive: true })
const combined = [
  'MAREA map viewer: third-party software',
  '',
  'The map viewer bundles the packages below. Each keeps its own licence, reproduced here.',
  'The MAREA code is under the MIT licence; its text, figures and data under CC BY 4.0 (see the repository).',
  '',
]
const missing = []
for (const [name, info] of [...packages.entries()].sort(([a], [b]) => a.localeCompare(b))) {
  const files = existsSync(info.dir) ? readdirSync(info.dir).filter((f) => LICENCE_FILE.test(f)).sort() : []
  const texts = files.map((f) => readFileSync(join(info.dir, f), 'utf8').replace(/\r\n/g, '\n').trim())
  if (texts.length === 0) missing.push(name)
  const header = `${name} ${info.version ?? ''} (${info.license ?? 'licence not stated'})`.replace(/\s+\(/, ' (')
  const body = texts.length ? texts.join('\n\n') : `No licence file in the package; package.json states: ${info.license ?? 'nothing'}.`
  writeFileSync(join(licDir, `${name.replace(/^@/, '').replace(/\//g, '__')}.txt`), `${header}\n\n${body}\n`)
  combined.push('='.repeat(100), header, '='.repeat(100), '', body, '')
}
writeFileSync(join(licDir, 'THIRD-PARTY-NOTICES.txt'), `${combined.join('\n')}\n`)
written.push(`licenses/ (${packages.size} packages)`)

// ---------------------------------------------------------------------------------------------------- 3: scan
const TEXT_EXT = new Set(['.html', '.js', '.mjs', '.css', '.json', '.geojson', '.txt', '.svg', '.webmanifest', '.map', '.xml'])
const ALL_FILES = [
  { what: 'a personal folder (Windows)', re: /[A-Za-z]:[\\/]{1,2}(?:Users|Documents and Settings)[\\/]/i },
  { what: 'a personal folder (Unix)', re: /\/(?:home|Users)\/[A-Za-z0-9._-]+\//i },
  { what: 'a mounted disk', re: /\/(?:media|mnt)\/[A-Za-z0-9._-]+\//i },
  { what: 'a temporary folder', re: /\/tmp\/[A-Za-z0-9._-]+/ },
  { what: 'a file URL', re: /file:\/\/\// },
]
const DATA_FILES = [
  { what: 'a Windows drive path', re: /(?<![A-Za-z0-9])[A-Za-z]:\\{1,2}[A-Za-z0-9_. -]+\\/ },
  {
    what: 'an IP address',
    re: /(?<![\d.])(?!(?:0\.0\.0\.0|127\.0\.0\.1)(?![\d.]))(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])/,
  },
]

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) walk(p, out)
    else out.push(p)
  }
  return out
}

const files = walk(outDir)
const hits = []
for (const f of files) {
  const ext = extname(f).toLowerCase()
  if (!TEXT_EXT.has(ext) || f.startsWith(licDir)) continue // licence texts are third-party prose
  const text = readFileSync(f, 'utf8')
  const checks = ext === '.js' || ext === '.mjs' ? ALL_FILES : [...ALL_FILES, ...DATA_FILES]
  for (const { what, re } of checks) {
    const m = re.exec(text)
    if (m) hits.push(`${rel(f)}: ${what}: "${m[0].slice(0, 60)}"`)
  }
}
if (hits.length) fail(`local paths or addresses in the build:\n  ${hits.join('\n  ')}`)

// ---------------------------------------------------------------------------------------------------- 4: sizes
const mb = (b) => `${(b / 1e6).toFixed(2)} MB`
const sizes = files.map((f) => ({ path: rel(f), bytes: statSync(f).size }))
const total = sizes.reduce((s, x) => s + x.bytes, 0)
const byTop = new Map()
for (const s of sizes) {
  const top = s.path.includes('/') ? `${s.path.split('/')[0]}/` : s.path
  byTop.set(top, (byTop.get(top) ?? 0) + s.bytes)
}
console.log(`postbuild: wrote ${written.join(', ')}`)
if (missing.length) console.log(`postbuild: no licence file in ${missing.join(', ')} (licence field copied instead)`)
console.log(`postbuild: ${files.length} files, ${mb(total)} in ${rel(outDir) || '.'} (output folder)`)
for (const [top, bytes] of [...byTop.entries()].sort((a, b) => b[1] - a[1])) {
  console.log(`  ${top.padEnd(28)} ${mb(bytes).padStart(10)}`)
}
const largest = [...sizes].sort((a, b) => b.bytes - a.bytes).slice(0, 5)
console.log(`  largest: ${largest.map((s) => `${s.path} ${mb(s.bytes)}`).join(', ')}`)
const tooBig = sizes.filter((s) => s.bytes > MAX_FILE)
if (tooBig.length) fail(`files over ${mb(MAX_FILE)}: ${tooBig.map((s) => s.path).join(', ')}`)
if (total > MAX_TOTAL) fail(`the site is ${mb(total)}, over the ${mb(MAX_TOTAL)} cap`)
console.log('postbuild: done')
