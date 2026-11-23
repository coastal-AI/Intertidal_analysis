// Build variables of the website, shared by vite.config.js and scripts/postbuild.mjs.
//
//   SITE_BASE        path the site is served under. Default /Intertidal_analysis/ (a GitHub Pages project site of
//                    the repository Intertidal_analysis); /<repository>/ for another repository; / for a custom domain.
//   SITE_PUBLIC_DIR  folder copied unchanged into the build (default: public). It replaces public/ as a whole.
//   SITE_OUT_DIR     build output folder (default: ../docs, the folder GitHub Pages serves). It is emptied first.
//   SITE_CACHE_DIR   Vite's dependency cache (default: node_modules/.vite).
//
// Relative folders are read from the site folder (the one that holds package.json).
import { parse, resolve, sep } from 'node:path'

export const DEFAULT_BASE = '/Intertidal_analysis/'

/** The base path with exactly one leading and one trailing slash ("" and "/" give "/"). */
export function normaliseBase(value) {
  const inner = String(value).trim().replace(/^\/+|\/+$/g, '')
  return inner ? `/${inner}/` : '/'
}

/**
 * Refuse a base path that is a folder rather than a URL path. Git Bash on Windows rewrites a variable value that
 * starts with "/" into a Windows folder before npm sees it (SITE_BASE=/ arrives as the Git install folder), which
 * would build a site whose every link is broken.
 */
export function assertUrlBase(value) {
  const v = String(value).trim()
  if (/^[A-Za-z]:/.test(v) || v.includes('\\')) {
    throw new Error(
      `SITE_BASE must be a URL path such as /Intertidal_analysis/ or /, not a folder (got "${v}"). ` +
        'In Git Bash, keep the shell from rewriting it: MSYS2_ENV_CONV_EXCL=SITE_BASE SITE_BASE=/... npm run build; ' +
        "in PowerShell: $env:SITE_BASE = '/...'; npm run build",
    )
  }
  return v
}

/** Refuse an output folder whose emptying would delete the site itself or a folder above it. */
export function assertSafeOutDir(outDir, siteDir) {
  const out = resolve(outDir)
  const site = resolve(siteDir)
  const isRoot = parse(out).root === out
  const isSiteOrAbove = site === out || site.startsWith(out.endsWith(sep) ? out : out + sep)
  if (isRoot || isSiteOrAbove) {
    throw new Error(`SITE_OUT_DIR must not be the site folder, a folder above it or a drive root (got ${out})`)
  }
  return out
}

/** The resolved build variables. */
export function siteEnv(siteDir, env = process.env) {
  const outDir = resolve(siteDir, env.SITE_OUT_DIR || '../docs')
  return {
    base: normaliseBase(assertUrlBase(env.SITE_BASE || DEFAULT_BASE)),
    publicDir: resolve(siteDir, env.SITE_PUBLIC_DIR || 'public'),
    outDir: assertSafeOutDir(outDir, siteDir),
    cacheDir: resolve(siteDir, env.SITE_CACHE_DIR || 'node_modules/.vite'),
  }
}
