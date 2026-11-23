/**
 * Texts of the map viewer's info panel, in one place, for the authors to edit: the poster's title, authors and
 * affiliations (CARTOGRASS 2026), the data sources with their licences, and how to cite the repository.
 */

export interface Author {
  name: string
  /** Numbers of the author's affiliations (see poster.affiliations). */
  affiliations: number[]
}

export const poster = {
  title: 'MAREA: a method for mapping intertidal topography using Sentinel-2 openEO data cubes',
  authors: [
    { name: 'Jorge Alías Álvarez', affiliations: [1] },
    { name: 'Fernando García-González', affiliations: [2] },
    { name: 'Pablo González', affiliations: [1] },
  ] as Author[],
  affiliations: [
    { id: 1, name: 'University of Oviedo, Spain' },
    { id: 2, name: 'Instituto Español de Oceanografía (IEO-CSIC), Spain' },
  ],
}

export const conference = {
  name: 'CARTOGRASS 2026',
  /** One line, above the title in the info panel. */
  line: 'CARTOGRASS 2026 · Cartagena, 24–26 November 2026',
}

/** A data source of the maps, with its licence. */
export interface DataSource {
  /** What it gives, e.g. "Images". */
  label: string
  /** The attribution, as the source asks for it. */
  credit: string
  licence: string
  /** The source's page. */
  href: string
  /** Shown only where the viewer offers its 3D view. */
  only3d?: boolean
}

export const dataSources: DataSource[] = [
  {
    label: 'Maps',
    credit: 'MAREA, by the authors',
    licence: 'CC BY 4.0',
    href: 'https://creativecommons.org/licenses/by/4.0/',
  },
  {
    label: 'Images',
    credit: 'Contains modified Copernicus Sentinel-2 data (2015–2026)',
    licence: 'free and open',
    href: 'https://dataspace.copernicus.eu/',
  },
  {
    label: 'Tide',
    credit: 'EOT20 (Hart-Davis et al. 2021)',
    licence: 'CC BY 4.0',
    href: 'https://doi.org/10.17882/79489',
  },
  {
    label: 'Sea level',
    credit: 'IOC Sea Level Station Monitoring Facility (VLIZ/IOC), doi:10.14284/482',
    licence: 'non-commercial use (CC BY-NC 4.0)',
    href: 'https://doi.org/10.14284/482',
  },
  {
    label: 'Orthophoto and base map',
    credit: 'PNOA and IGN base map, © Instituto Geográfico Nacional',
    licence: 'CC BY 4.0 scne.es',
    href: 'https://www.ign.es',
  },
  {
    label: '3D terrain',
    credit: 'Mapterhorn (IGN; Copernicus GLO-30), with the MAREA heights',
    licence: 'open data',
    href: 'https://mapterhorn.com/attribution',
    only3d: true,
  },
]

/** How to cite the repository (as in CITATION.cff), with its address. */
export function repositoryCitation(repoUrl: string): string {
  return (
    'Alías Álvarez, J., García-González, F., González, P. (2026). MAREA: a method for mapping intertidal topography ' +
    `using Sentinel-2 openEO data cubes (CARTOGRASS 2026 companion repository), version cartograss26. ${repoUrl}`
  )
}
