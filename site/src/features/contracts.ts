/**
 * Interface between the app shell (App.tsx) and the map viewer, the whole site.
 *
 * src/features/explorer/index.tsx   default export: (props: ExplorerProps) => JSX, lazy-loaded by App.tsx
 *   Rendered in a full-screen box (100dvh, no page scroll) at the site's root; the component fills it
 *   (h-full w-full) and keeps its state in the URL hash, so views can be shared:
 *     #site=<SiteId>&layer=<id>&base=photo|map&3d=0|1
 *   Data, relative to the data folder (dataUrl() in @/lib/assets):
 *     sites/index.json          the site list, contract "cg26-sites/1" (types and loader in @/lib/sites)
 *     sites/<id>/manifest.json  the site's layers, legends and files (format owned by the viewer)
 */
import type { Enable3D } from '@/config'
import type { SiteId } from '@/lib/sites'

export interface ExplorerProps {
  /** Site shown when the URL hash names none. */
  defaultSite: SiteId
  /** Whether a 2D/3D toggle may be shown: everywhere ("on"), on wide screens ("desktop") or never ("off"). */
  enable3d: Enable3D
}
