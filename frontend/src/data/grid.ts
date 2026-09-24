import proj4 from 'proj4'

// The common 100 m analysis grid (thesis Section 4.3.2), UTM Zone 51N.
export const GRID = {
  cols: 242,
  rows: 490,
  cell: 100,
  x0: 276_700, // upper-left easting
  y0: 1_635_100, // upper-left northing
} as const

export const CELL_COUNT = GRID.cols * GRID.rows

const UTM51 = '+proj=utm +zone=51 +datum=WGS84 +units=m +no_defs'
const toLngLat = proj4(UTM51, 'WGS84')

export type LngLat = [number, number]

export function utmToLngLat(x: number, y: number): LngLat {
  return toLngLat.forward([x, y]) as LngLat
}

export function lngLatToUtm(lng: number, lat: number): [number, number] {
  return toLngLat.inverse([lng, lat]) as [number, number]
}

const xMax = GRID.x0 + GRID.cols * GRID.cell
const yMin = GRID.y0 - GRID.rows * GRID.cell

/** Image corners in MapLibre order: top-left, top-right, bottom-right, bottom-left. */
export const GRID_CORNERS: [LngLat, LngLat, LngLat, LngLat] = [
  utmToLngLat(GRID.x0, GRID.y0),
  utmToLngLat(xMax, GRID.y0),
  utmToLngLat(xMax, yMin),
  utmToLngLat(GRID.x0, yMin),
]

export const GRID_BOUNDS: [LngLat, LngLat] = [
  [Math.min(GRID_CORNERS[0][0], GRID_CORNERS[3][0]), Math.min(GRID_CORNERS[2][1], GRID_CORNERS[3][1])],
  [Math.max(GRID_CORNERS[1][0], GRID_CORNERS[2][0]), Math.max(GRID_CORNERS[0][1], GRID_CORNERS[1][1])],
]

export interface CellRef {
  col: number
  row: number
  index: number
}

export function cellAt(lng: number, lat: number): CellRef | null {
  const [x, y] = lngLatToUtm(lng, lat)
  const col = Math.floor((x - GRID.x0) / GRID.cell)
  const row = Math.floor((GRID.y0 - y) / GRID.cell)
  if (col < 0 || row < 0 || col >= GRID.cols || row >= GRID.rows) return null
  return { col, row, index: row * GRID.cols + col }
}

export function cellFromIndex(index: number): CellRef {
  return { index, col: index % GRID.cols, row: Math.floor(index / GRID.cols) }
}

export function cellCenter(c: CellRef): LngLat {
  return utmToLngLat(GRID.x0 + (c.col + 0.5) * GRID.cell, GRID.y0 - (c.row + 0.5) * GRID.cell)
}

export function cellPolygon(c: CellRef): LngLat[] {
  const x = GRID.x0 + c.col * GRID.cell
  const y = GRID.y0 - c.row * GRID.cell
  const d = GRID.cell
  return [
    utmToLngLat(x, y),
    utmToLngLat(x + d, y),
    utmToLngLat(x + d, y - d),
    utmToLngLat(x, y - d),
    utmToLngLat(x, y),
  ]
}

/** Planar UTM distance in meters between two lng/lat points. */
export function metersBetween(a: LngLat, b: LngLat): number {
  const [ax, ay] = lngLatToUtm(a[0], a[1])
  const [bx, by] = lngLatToUtm(b[0], b[1])
  return Math.hypot(ax - bx, ay - by)
}
