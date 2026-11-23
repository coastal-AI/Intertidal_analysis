/**
 * Colours and dashes of the map's lines, shared by the map and its legend (no MapLibre import: the tests use it).
 */

/** Tide lines: the poster's ink. */
export const CONTOUR_COLOUR = '#1F2A30'
/** Lines of equal delay: a dark tone of the delay ramp. */
export const EQUAL_DELAY_COLOUR = '#804C22'
/** Dash of the lines of equal delay, in line widths. */
export const EQUAL_DELAY_DASH = [2, 1.5]

/** Tide lines: mean sea level solid, neap levels dashed, spring and extreme levels dotted (dashes in line widths). */
export const CONTOUR_STYLES: readonly { id: string; datums: readonly string[]; dash: number[] | null }[] = [
  { id: 'contours-solid', datums: ['MSL'], dash: null },
  { id: 'contours-dash', datums: ['MLWN', 'MHWN'], dash: [3, 2] },
  { id: 'contours-dot', datums: ['LAT', 'MLWS', 'MHWS', 'HAT'], dash: [1, 1.5] },
]

/** The dash of a tide level's line (null = solid). */
export function contourDash(datum: string): number[] | null {
  return CONTOUR_STYLES.find((s) => s.datums.includes(datum))?.dash ?? null
}
