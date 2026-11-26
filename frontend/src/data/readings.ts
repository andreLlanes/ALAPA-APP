// Hourly station readings around a map's origin (readings.json from O3/spatial/export.py): the
// lookback the model saw before the bulletin hour and, for a past forecast, what the stations
// measured during the forecast hours. Only stations measure; there is no reading between them.

import { categoryIndex } from './aqi'
import { HORIZON, type ForecastField, type ModelId } from './forecast'
import { cellAt, type LngLat } from './grid'

export interface StationReadings {
  id: string
  name: string
  at: LngLat
  reference: boolean
  /** Its forecast went into the map. The map never saw the others, so they test it independently. */
  inMap: boolean
  /** Index of the grid cell the station sits in. */
  cell: number
  /** µg/m³ per hour, hour k = lead k - lookback; NaN where there is no reading. */
  pm25: Float32Array
}

export interface Readings {
  /** Hours before the bulletin hour included (the model's lookback). */
  lookback: number
  stations: StationReadings[]
  /** Some forecast hours have readings: the map is a past forecast that can be checked. */
  backtest: boolean
}

export interface ReadingsFile {
  start_utc: string
  hours: number
  lookback: number
  stations: { id: string; lat: number; lon: number; reference: boolean; in_map: boolean; pm25: (number | null)[] }[]
}

export const stationName = (id: string) => `Station ${id.replace(/^openaq:/, '')}`

export function parseReadings(file: ReadingsFile, issued: number): Readings {
  if (Date.parse(file.start_utc) + file.lookback * 3_600_000 !== issued || file.hours !== file.lookback + HORIZON) {
    throw new Error('the station readings do not line up with the forecast hours')
  }
  const stations: StationReadings[] = []
  let backtest = false
  for (const s of file.stations) {
    const cell = cellAt(s.lon, s.lat)
    if (!cell) continue
    const pm25 = Float32Array.from(s.pm25, (v) => (v === null ? NaN : v))
    for (let k = file.lookback; k < file.hours && !backtest; k++) backtest = !Number.isNaN(pm25[k])
    stations.push({ id: s.id, name: stationName(s.id), at: [s.lon, s.lat], reference: s.reference, inMap: s.in_map, cell: cell.index, pm25 })
  }
  return { lookback: file.lookback, stations, backtest }
}

/** The reading at a lead hour (negative leads are the lookback); NaN when there is none. */
export function readingAt(r: Readings, s: StationReadings, lead: number) {
  const k = lead + r.lookback
  return k >= 0 && k < s.pm25.length ? s.pm25[k] : NaN
}

export interface Score {
  stations: number
  /** Station-hours with both a reading and a forecast. */
  hours: number
  rmse: number
  mae: number
  /** Mean of forecast minus reading: positive when the forecast ran high. */
  bias: number
  /** Share of hours where forecast and reading fall in the same AQI level. */
  sameLevel: number
}

/** Forecast against readings over the 72 forecast hours, at the cells the stations sit in. */
export function score(field: ForecastField, model: ModelId, stations: StationReadings[]): Score {
  let n = 0
  let se = 0
  let ae = 0
  let e = 0
  let same = 0
  let used = 0
  for (const s of stations) {
    let any = false
    for (let h = 0; h < HORIZON; h++) {
      const obs = field.readings ? readingAt(field.readings, s, h) : NaN
      const f = field.value(s.cell, h, model)
      if (!Number.isFinite(obs) || !Number.isFinite(f)) continue
      const d = f - obs
      n++
      se += d * d
      ae += Math.abs(d)
      e += d
      if (categoryIndex(f) === categoryIndex(obs)) same++
      any = true
    }
    if (any) used++
  }
  return {
    stations: used,
    hours: n,
    rmse: n ? Math.sqrt(se / n) : NaN,
    mae: n ? ae / n : NaN,
    bias: n ? e / n : NaN,
    sameLevel: n ? same / n : NaN,
  }
}
