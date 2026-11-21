// Forecast maps exported by O3/spatial/export.py: for each model, PM2.5 and its kriging standard
// deviation for 72 hours on the 100 m grid, one byte per cell and hour, with a manifest that holds
// the clock, the stations behind the maps, each model's encodings and test score, and which model
// to show first. A version 1 manifest (a single map, from before several models were exported)
// still loads, as one model.

import {
  ForecastField,
  HORIZON,
  MODELS,
  bulletinNumber,
  type Clock,
  type ModelId,
  type ModelInfo,
  type Provenance,
  type Station,
} from './forecast'
import { CELL_COUNT, GRID } from './grid'
import { parseReadings, stationName, type Readings, type ReadingsFile } from './readings'

interface PmEncoding {
  file: string
  encoding: 'sqrt'
  max: number
  nodata: number
}

interface SdEncoding {
  file: string
  encoding: 'linear'
  scale: number[]
  nodata: number
  includes: string
}

/** One model's map in the manifest. */
interface ModelMap {
  /** Null only in a version 1 file that did not record its model. */
  id: string | null
  variant: string | null
  seeds: number[]
  source: string
  pm25: PmEncoding
  sd: SdEncoding
  variogram?: unknown
  /** The model's leave-one-station-out RMSE over all folds, when O3's table had it. */
  loso?: { rmse: number; folds: number; seeds: number } | null
}

interface Common {
  grid: { epsg: number; x0: number; y0: number; cell: number; rows: number; cols: number }
  origin_utc: string
  issued_utc: string
  data_as_of_utc: string
  horizon: number
  stations: { id: string; lat: number; lon: number; reference: boolean }[]
  interval: { coverage: number; z: number }
  /** Absent from maps exported before station readings were added. */
  readings?: { file: string; stations: number; forecast_hour_readings: number }
}

interface ManifestV1 extends Common {
  version: 1
  source: string
  model: { id: string | null; variant: string | null; seeds: number[] }
  pm25: PmEncoding
  sd: SdEncoding
  variogram?: unknown
}

export interface Manifest extends Common {
  version: 2
  default_model: string
  /** Why that model comes first: chosen by hand, lowest leave-one-station-out RMSE, or first given. */
  default_by: 'choice' | 'loso_rmse' | 'order'
  models: ModelMap[]
}

/** The id of a map whose file does not record its model. */
const UNKNOWN = 'map'

/** A version 1 manifest as a version 2 one with a single model. */
function upgrade(m: ManifestV1 | Manifest): Manifest {
  if (m.version === 2) return m
  const { model, pm25, sd, source, variogram, ...common } = m
  return {
    ...common,
    version: 2,
    default_model: model.id ?? UNKNOWN,
    default_by: 'order',
    models: [{ id: model.id, variant: model.variant, seeds: model.seeds, source, pm25, sd, variogram }],
  }
}

/** Where the exported map lives: VITE_FORECAST_URL, or data/ next to the app. */
const DATA_URL = (import.meta.env.VITE_FORECAST_URL as string | undefined) ?? `${import.meta.env.BASE_URL}data/`

/** One model's decoded map. */
interface Layer {
  info: ModelMap
  pm: Uint8Array
  sd: Uint8Array
  /** Byte to µg/m³. */
  lut: Float32Array
  sdScale: Float32Array
  conf: Float32Array
  note: string
}

export class GridField extends ForecastField {
  readonly clock: Clock
  readonly stations: Station[]
  readonly models: ModelInfo[]
  readonly provenance: Provenance = { demo: false, calibratedRange: false }
  readonly readings: Readings | null
  /** The maps' own record, for the details panel. */
  readonly manifest: Manifest
  private readonly layers = new Map<ModelId, Layer>()
  private readonly first: Layer
  private readonly firstId: ModelId
  private readonly z: number

  constructor(manifest: Manifest, grids: { pm: Uint8Array; sd: Uint8Array }[], readings: ReadingsFile | null) {
    super()
    this.manifest = manifest
    this.z = manifest.interval.z
    const issued = Date.parse(manifest.issued_utc)
    this.clock = { issued, dataAsOf: Date.parse(manifest.data_as_of_utc), bulletinNo: bulletinNumber(issued) }
    this.readings = readings ? parseReadings(readings, issued) : null
    this.stations = manifest.stations.map((s) => ({
      id: s.id,
      name: stationName(s.id),
      at: [s.lon, s.lat],
      reference: s.reference,
    }))
    this.models = manifest.models.map(
      (m) => MODELS.find((k) => k.id === m.id) ?? { id: m.id ?? UNKNOWN, name: 'Map', full: 'The map file does not record its model' },
    )
    manifest.models.forEach((info, k) => {
      const id = this.models[k].id
      const known = MODELS.find((m) => m.id === info.id)
      this.layers.set(id, makeLayer(info, grids[k], noteFor(info, known, this.stations.length, manifest, id === manifest.default_model)))
    })
    this.firstId = this.layers.has(manifest.default_model) ? manifest.default_model : this.models[0].id
    this.first = this.layers.get(this.firstId)!
    this.buildDistances()
  }

  get defaultModel() {
    return this.firstId
  }

  private layer(model: ModelId) {
    return this.layers.get(model) ?? this.first
  }

  /** Every model's map blanks the same cells; the export checks it. */
  isWater(i: number) {
    return this.first.pm[i] === this.first.info.pm25.nodata
  }

  value(i: number, lead: number, model: ModelId): number {
    if (lead < 0 || lead >= HORIZON) return NaN
    const l = this.layer(model)
    return l.lut[l.pm[lead * CELL_COUNT + i]]
  }

  /** The kriging part only: the forecast's own error is not in the exported map yet. */
  halfWidth(i: number, lead: number, model: ModelId): number {
    if (lead < 0 || lead >= HORIZON) return NaN
    const l = this.layer(model)
    const s = l.sd[lead * CELL_COUNT + i]
    return s === l.info.sd.nodata ? NaN : (this.z * s * l.sdScale[lead]) / 254
  }

  confidence(i: number, model: ModelId) {
    return this.layer(model).conf[i]
  }

  modelNote(model: ModelId) {
    return this.layer(model).note
  }
}

function makeLayer(info: ModelMap, grid: { pm: Uint8Array; sd: Uint8Array }, note: string): Layer {
  const lut = new Float32Array(256)
  for (let c = 0; c < 255; c++) lut[c] = info.pm25.max * (c / 254) ** 2
  lut[255] = NaN
  // Kriging variance keeps one spatial shape every hour, so the first hour ranks the cells.
  const { sd } = grid
  const nodata = info.sd.nodata
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < CELL_COUNT; i++) {
    const s = sd[i]
    if (s === nodata) continue
    if (s < lo) lo = s
    if (s > hi) hi = s
  }
  const conf = new Float32Array(CELL_COUNT)
  for (let i = 0; i < CELL_COUNT; i++) {
    const s = sd[i]
    conf[i] = s === nodata || hi <= lo ? 0 : 1 - (s - lo) / (hi - lo)
  }
  return { info, pm: grid.pm, sd, lut, sdScale: Float32Array.from(info.sd.scale), conf, note }
}

/** The line under the model choice: what made this map and how it scored when tested. */
function noteFor(info: ModelMap, known: ModelInfo | undefined, stations: number, manifest: Manifest, isDefault: boolean) {
  const variant = info.variant && info.variant !== 'none' ? `, ${info.variant}` : ''
  const seeds = info.seeds.length
  let note =
    `${known ? `${known.full} (${known.name}${variant})` : 'Model not recorded in the map file'}` +
    `${seeds ? `, mean of ${seeds} seeds` : ''}, regression-kriged to 100 m from ${stations} stations.`
  if (info.loso) {
    note +=
      ' Tested by leaving out one station at a time and mapping it from the others: typical miss ' +
      `${info.loso.rmse.toFixed(1)} µg/m³ (RMSE over ${info.loso.folds} stations).`
  }
  if (isDefault && manifest.default_by === 'loso_rmse' && manifest.models.length > 1) {
    note += ` Shown first because that miss is the lowest of the ${manifest.models.length} models.`
  }
  return note
}

/** Fetch a byte grid, inflating it when the server sent the gzip file as is. */
async function fetchGrid(url: string): Promise<Uint8Array> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`)
  let bytes = new Uint8Array(await res.arrayBuffer())
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))
    bytes = new Uint8Array(await new Response(stream).arrayBuffer())
  }
  if (bytes.length !== HORIZON * CELL_COUNT) {
    throw new Error(`${url}: ${bytes.length} bytes, expected ${HORIZON * CELL_COUNT}`)
  }
  return bytes
}

/** Load the exported forecast maps; null when none is published. */
export async function loadForecast(base = DATA_URL): Promise<GridField | null> {
  const res = await fetch(`${base}manifest.json`, { cache: 'no-cache' })
  // A dev server answers a missing file with the app's HTML page, so check the type too.
  if (!res.ok || !res.headers.get('content-type')?.includes('json')) return null
  const raw = (await res.json()) as ManifestV1 | Manifest
  const version = (raw as { version?: unknown }).version
  if (version !== 1 && version !== 2) throw new Error(`the forecast map is format ${String(version)}, newer than this app reads`)
  const manifest = upgrade(raw)
  const g = manifest.grid
  if (manifest.horizon !== HORIZON || g.cols !== GRID.cols || g.rows !== GRID.rows
      || g.cell !== GRID.cell || g.x0 !== GRID.x0 || g.y0 !== GRID.y0) {
    throw new Error('the forecast map is not on the app grid')
  }
  if (!manifest.models.length) throw new Error('the forecast map file lists no models')
  const [grids, readings] = await Promise.all([
    Promise.all(manifest.models.map(async (m) => {
      const [pm, sd] = await Promise.all([fetchGrid(base + m.pm25.file), fetchGrid(base + m.sd.file)])
      return { pm, sd }
    })),
    manifest.readings ? fetchReadings(base + manifest.readings.file) : null,
  ])
  return new GridField(manifest, grids, readings)
}

/** The station readings are extra: without them the map still shows, only the checks are missing. */
async function fetchReadings(url: string): Promise<ReadingsFile | null> {
  try {
    const res = await fetch(url, { cache: 'no-cache' })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return (await res.json()) as ReadingsFile
  } catch (err) {
    console.warn(`${url}: station readings not loaded`, err)
    return null
  }
}
