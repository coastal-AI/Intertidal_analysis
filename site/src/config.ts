/**
 * Build-time configuration of the site, read from Vite's environment (see README.md, "Build variables").
 *
 *   BASE_URL        the base path, set by SITE_BASE (default "/Intertidal_analysis/")
 *   VITE_REPO_URL   the repository page the site links to, a GitHub "tree" URL (default: the CARTOGRASS26 branch)
 *   VITE_ENABLE_3D  where the map viewer may offer its 3D view: "on" | "desktop" | "off" (default "desktop")
 */

/** Where the map viewer may show its 2D/3D toggle. */
export type Enable3D = 'on' | 'desktop' | 'off'

export const DEFAULT_REPO_URL = 'https://github.com/coastal-AI/Intertidal_analysis/tree/CARTOGRASS26'

/** Read VITE_ENABLE_3D; anything but "on" or "off" means "desktop". */
export function parseEnable3d(value: string | undefined): Enable3D {
  const v = (value ?? '').trim().toLowerCase()
  return v === 'on' || v === 'off' ? v : 'desktop'
}

/** Owner, repository and branch of a GitHub repository URL. */
export interface RepoRef {
  owner: string
  name: string
  /** Branch or tag; "" when the URL names none. */
  ref: string
}

/** Parse https://github.com/<owner>/<name>[/tree/<ref>]; null for any other URL. */
export function parseRepoUrl(url: string): RepoRef | null {
  const m = /^https:\/\/github\.com\/([^/\s]+)\/([^/\s]+?)(?:\.git)?(?:\/tree\/([^\s]+?))?\/?$/.exec(url.trim())
  if (!m) return null
  return { owner: m[1], name: m[2], ref: m[3] ?? '' }
}

const repoUrl = (import.meta.env.VITE_REPO_URL || DEFAULT_REPO_URL).trim().replace(/\/+$/, '')

export const config = {
  /** Base path with leading and trailing slash, e.g. "/Intertidal_analysis/". */
  base: import.meta.env.BASE_URL,
  /** The repository page the site links to. */
  repoUrl,
  /** Where the map viewer may show its 3D toggle. */
  enable3d: parseEnable3d(import.meta.env.VITE_ENABLE_3D),
} as const

/** Link to a file of the repository at its branch (GitHub "blob" view). */
export function repoFile(path: string, url: string = config.repoUrl): string {
  const clean = path.replace(/^\/+/, '')
  const r = parseRepoUrl(url)
  if (!r) return `${url}/${clean}`
  const ref = r.ref || 'HEAD'
  return `https://github.com/${r.owner}/${r.name}/blob/${ref}/${clean}`
}
