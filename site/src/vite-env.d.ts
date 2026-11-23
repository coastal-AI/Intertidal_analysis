/// <reference types="vite/client" />

// Client variables of the site (all optional; see src/config.ts and README.md).
interface ImportMetaEnv {
  /** Repository page the site links to, a GitHub "tree" URL (default: the CARTOGRASS26 branch). */
  readonly VITE_REPO_URL?: string
  /** Where the map explorer may offer its 3D view: "on" | "desktop" | "off" (default "desktop"). */
  readonly VITE_ENABLE_3D?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
