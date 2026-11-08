// Synthetic stand-in for the ALAPA forecast API. Every number produced here is
// demo data shaped to look like Metro Manila; the UI labels it as such.

import { AQI, categoryIndex } from './aqi'
import { CELL_COUNT, GRID, lngLatToUtm, utmToLngLat, type LngLat } from './grid'
import { CITIES, CORRIDORS, GREEN, HOTSPOTS, WATER } from './places'

export const HORIZON = 72
export type ModelId = 'gnn' | 'lstm' | 'gbt'
export type Scenario = 'normal' | 'clean' | 'severe'

export const MODELS: { id: ModelId; name: string; full: string }[] = [
  { id: 'gnn', name: 'GNN', full: 'Graph neural network' },
  { id: 'lstm', name: 'LSTM', full: 'Long short-term memory' },
  { id: 'gbt', name: 'GBT', full: 'Gradient-boosted trees' },
]

const MODEL_SHAPE: Record<ModelId, { bias: number; wobble: number; spread: number; phase: number }> = {
  gnn: { bias: 1, wobble: 0.04, spread: 1, phase: 0 },
  lstm: { bias: 1.04, wobble: 0.03, spread: 1.12, phase: 1.7 },
  gbt: { bias: 0.95, wobble: 0.07, spread: 1.22, phase: 3.1 },
}

// ---------------------------------------------------------------- utilities

function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gauss(x: number, mu: number, s: number) {
  const d = (x - mu) / s
  return Math.exp(-0.5 * d * d)
}

/** Gaussian on a 24-hour clock. */
function clockBump(hour: number, mu: number, s: number) {
  let d = Math.abs(hour - mu)
  d = Math.min(d, 24 - d)
  return Math.exp(-0.5 * (d / s) ** 2)
}

function pointInPolygon(x: number, y: number, poly: [number, number][]) {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

function segDistance(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
  const dx = bx - ax
  const dy = by - ay
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
}

const utm = (p: LngLat) => lngLatToUtm(p[0], p[1])

// ---------------------------------------------------------------- clock

const MANILA_OFFSET_H = 8

export interface Clock {
  /** Top of the issue hour, epoch ms. Lead hour 0 is this hour. */
  issued: number
  /** When the newest station reading arrived, epoch ms. */
  dataAsOf: number
  bulletinNo: number
  episodePeak: number
  rainStart: number
}

export function makeClock(now = Date.now(), staleMinutes = 0): Clock {
  const issued = Math.floor(now / 3_600_000) * 3_600_000
  const localHour = (new Date(issued).getUTCHours() + MANILA_OFFSET_H) % 24
  const toMidnight = 24 - localHour
  return {
    issued,
    dataAsOf: issued - (staleMinutes > 0 ? staleMinutes : 20) * 60_000,
    // Numbering rule is undecided; this counts refreshes since the demo epoch.
    bulletinNo: Math.floor((issued - Date.UTC(2026, 8, 1)) / 3_600_000) % 10_000,
    episodePeak: toMidnight + 21,
    rainStart: toMidnight + 24 + 14,
  }
}

export function localHourAt(clock: Clock, lead: number) {
  return (new Date(clock.issued + lead * 3_600_000).getUTCHours() + MANILA_OFFSET_H) % 24
}

// ---------------------------------------------------------------- stations

export interface Station {
  id: string
  name: string
  at: LngLat
  reference: boolean
}

// ---------------------------------------------------------------- the field

export class ForecastField {
  readonly clock: Clock
  readonly scenario: Scenario
  readonly stations: Station[]
  /** Spatial base concentration, µg/m³. NaN marks water. */
  readonly base = new Float32Array(CELL_COUNT)
  /** Traffic exposure 0..1, drives the rush-hour swing. */
  readonly traffic = new Float32Array(CELL_COUNT)
  /** Meters from the nearest station. */
  readonly stationDistance = new Float32Array(CELL_COUNT)

  private metroShareCache = new Map<ModelId, Float32Array>()
  private spreadCache = new Map<ModelId, Float32Array>()

  constructor(clock: Clock, scenario: Scenario = 'normal') {
    this.clock = clock
    this.scenario = scenario
    this.buildBase()
    this.stations = this.placeStations()
    this.buildDistances()
  }

  isWater(i: number) {
    return Number.isNaN(this.base[i])
  }

  private buildBase() {
    const water = WATER.map((poly) => poly.map(utm))
    const roads = CORRIDORS.map((c) => ({ w: c.weight, pts: c.line.map(utm) }))
    const hot = HOTSPOTS.map((h) => ({ ...h, xy: utm(h.at) }))
    const green = GREEN.map((g) => ({ ...g, xy: utm(g.at) }))
    const rnd = mulberry32(7)
    const ph = [rnd() * 6, rnd() * 6, rnd() * 6, rnd() * 6]

    for (let row = 0; row < GRID.rows; row++) {
      const y = GRID.y0 - (row + 0.5) * GRID.cell
      for (let col = 0; col < GRID.cols; col++) {
        const x = GRID.x0 + (col + 0.5) * GRID.cell
        const i = row * GRID.cols + col
        if (water.some((poly) => pointInPolygon(x, y, poly))) {
          this.base[i] = NaN
          continue
        }
        let road = 0
        for (const r of roads) {
          let d = Infinity
          for (let k = 1; k < r.pts.length; k++) {
            d = Math.min(d, segDistance(x, y, r.pts[k - 1][0], r.pts[k - 1][1], r.pts[k][0], r.pts[k][1]))
          }
          road = Math.max(road, r.w * Math.exp(-d / 420))
        }
        let v = 13 + 17 * road
        for (const h of hot) v += h.amp * Math.exp(-(((x - h.xy[0]) ** 2 + (y - h.xy[1]) ** 2) / (2 * h.radius ** 2)))
        for (const g of green) v += g.amp * Math.exp(-(((x - g.xy[0]) ** 2 + (y - g.xy[1]) ** 2) / (2 * g.radius ** 2)))
        // Density falls off toward the rural edges of the bounding box.
        const edge = Math.min(1, Math.hypot((x - 289_000) / 16_000, (y - 1_613_000) / 26_000))
        v -= 4 * edge * edge
        v += 1.6 * Math.sin(x / 1900 + ph[0]) * Math.cos(y / 2300 + ph[1]) + 1.1 * Math.sin((x + y) / 1300 + ph[2])
        v += 0.8 * Math.sin(x / 520 + ph[3]) * Math.sin(y / 610)
        this.base[i] = Math.max(4, v)
        this.traffic[i] = road
      }
    }
  }

  private placeStations(): Station[] {
    const rnd = mulberry32(2026)
    const anchors = [
      { at: CITIES[3].at, w: 5, s: 1600 }, // Makati
      { at: CITIES[0].at, w: 6, s: 3200 }, // Quezon City
      { at: CITIES[1].at, w: 5, s: 2000 }, // Manila
      { at: CITIES[4].at, w: 3, s: 2000 }, // Pasig
      { at: CITIES[5].at, w: 3, s: 2000 }, // Taguig
      { at: CITIES[12].at, w: 2, s: 1200 }, // Mandaluyong
      { at: CITIES[6].at, w: 1.5, s: 2400 }, // Parañaque
      { at: CITIES[2].at, w: 1.5, s: 2000 }, // Caloocan
      { at: CITIES[7].at, w: 1, s: 2400 }, // Valenzuela
      { at: CITIES[10].at, w: 1, s: 2200 }, // Marikina
      { at: CITIES[9].at, w: 0.8, s: 2600 }, // Muntinlupa
      { at: CITIES[8].at, w: 0.7, s: 2000 }, // Las Piñas
    ]
    const total = anchors.reduce((s, a) => s + a.w, 0)
    const out: Station[] = []
    let guard = 0
    while (out.length < 69 && guard++ < 5000) {
      let pick = rnd() * total
      const a = anchors.find((an) => (pick -= an.w) <= 0) ?? anchors[0]
      const [ax, ay] = utm(a.at)
      const r = Math.sqrt(-2 * Math.log(rnd() + 1e-9))
      const th = rnd() * Math.PI * 2
      const x = ax + r * Math.cos(th) * a.s
      const y = ay + r * Math.sin(th) * a.s
      const col = Math.floor((x - GRID.x0) / GRID.cell)
      const row = Math.floor((GRID.y0 - y) / GRID.cell)
      if (col < 0 || row < 0 || col >= GRID.cols || row >= GRID.rows) continue
      if (this.isWater(row * GRID.cols + col)) continue
      const n = out.length + 1
      out.push({
        id: `openaq-demo-${String(n).padStart(3, '0')}`,
        name: `Demo station ${String(n).padStart(2, '0')}`,
        at: utmToLngLat(x, y),
        reference: n === 7 || n === 31,
      })
    }
    return out
  }

  private buildDistances() {
    const pts = this.stations.map((s) => utm(s.at))
    for (let row = 0; row < GRID.rows; row++) {
      const y = GRID.y0 - (row + 0.5) * GRID.cell
      for (let col = 0; col < GRID.cols; col++) {
        const x = GRID.x0 + (col + 0.5) * GRID.cell
        let d = Infinity
        for (const p of pts) {
          const dd = (x - p[0]) ** 2 + (y - p[1]) ** 2
          if (dd < d) d = dd
        }
        this.stationDistance[row * GRID.cols + col] = Math.sqrt(d)
      }
    }
  }

  // ------------------------------------------------------------ time shape

  /** Metro-wide multiplier for a lead hour (negative leads are the recent past). */
  private timeFactor(lead: number, traffic: number) {
    const hour = localHourAt(this.clock, lead)
    const rush = 0.42 * clockBump(hour, 7.5, 1.5) + 0.36 * clockBump(hour, 19, 1.8)
    const night = 0.16 * clockBump(hour, 1, 3)
    const midday = -0.14 * clockBump(hour, 13.5, 2.5)
    const diurnal = 0.84 + (0.35 + 0.65 * traffic) * rush + night + midday
    const episode = 1 + 0.95 * gauss(lead, this.clock.episodePeak, 14)
    const rain = 1 - 0.34 / (1 + Math.exp(-(lead - this.clock.rainStart) / 1.8))
    const scale = this.scenario === 'clean' ? 0.5 : this.scenario === 'severe' ? 1.55 : 1
    return diurnal * episode * rain * scale
  }

  value(i: number, lead: number, model: ModelId): number {
    const b = this.base[i]
    if (Number.isNaN(b)) return NaN
    const m = MODEL_SHAPE[model]
    const col = i % GRID.cols
    const row = (i - col) / GRID.cols
    const wobble = 1 + m.wobble * Math.sin(col * 0.09 + row * 0.05 + lead * 0.21 + m.phase)
    return b * this.timeFactor(lead, this.traffic[i]) * m.bias * wobble
  }

  /** Half-width of the prediction interval, µg/m³. Level is undecided. */
  halfWidth(i: number, lead: number, model: ModelId, v = this.value(i, lead, model)) {
    const far = Math.min(1, this.stationDistance[i] / 4000)
    return v * (0.09 + 0.0042 * Math.max(0, lead)) * MODEL_SHAPE[model].spread + 3.2 * far
  }

  /** Relative map confidence 0..1 from the spatialization step; not an error bound. */
  confidence(i: number) {
    return Math.max(0, 1 - this.stationDistance[i] / 4500)
  }

  fillHour(lead: number, model: ModelId, out: Float32Array) {
    for (let i = 0; i < CELL_COUNT; i++) out[i] = this.value(i, lead, model)
    return out
  }

  series(i: number, model: ModelId) {
    const v = new Float32Array(HORIZON)
    const lo = new Float32Array(HORIZON)
    const hi = new Float32Array(HORIZON)
    for (let h = 0; h < HORIZON; h++) {
      v[h] = this.value(i, h, model)
      const w = this.halfWidth(i, h, model, v[h])
      lo[h] = Math.max(0, v[h] - w)
      hi[h] = v[h] + w
    }
    return { v, lo, hi }
  }

  /** Change against three hours earlier. */
  trend(i: number, lead: number, model: ModelId): 'rising' | 'easing' | 'steady' {
    const now = this.value(i, lead, model)
    const before = this.value(i, lead - 3, model)
    const r = (now - before) / before
    return r > 0.08 ? 'rising' : r < -0.08 ? 'easing' : 'steady'
  }

  /** Share of Metro land in each AQI category, per hour: [hour * 6 + category]. */
  metroShare(model: ModelId): Float32Array {
    const hit = this.metroShareCache.get(model)
    if (hit) return hit
    const share = new Float32Array(HORIZON * AQI.length)
    const step = 4
    for (let h = 0; h < HORIZON; h++) {
      let n = 0
      for (let row = 0; row < GRID.rows; row += step) {
        for (let col = 0; col < GRID.cols; col += step) {
          const v = this.value(row * GRID.cols + col, h, model)
          if (Number.isNaN(v)) continue
          share[h * AQI.length + categoryIndex(v)]++
          n++
        }
      }
      for (let k = 0; k < AQI.length; k++) share[h * AQI.length + k] /= n
    }
    this.metroShareCache.set(model, share)
    return share
  }

  /**
   * Median interval half-width across the area, per hour, µg/m³. The outlook
   * cone is drawn from this, so its shape reports the forecast's uncertainty
   * rather than illustrating it.
   */
  intervalSpread(model: ModelId): Float32Array {
    const hit = this.spreadCache.get(model)
    if (hit) return hit
    const out = new Float32Array(HORIZON)
    const step = 8
    const buf: number[] = []
    for (let h = 0; h < HORIZON; h++) {
      buf.length = 0
      for (let row = 0; row < GRID.rows; row += step) {
        for (let col = 0; col < GRID.cols; col += step) {
          const i = row * GRID.cols + col
          const v = this.value(i, h, model)
          if (!Number.isNaN(v)) buf.push(this.halfWidth(i, h, model, v))
        }
      }
      buf.sort((a, b) => a - b)
      out[h] = buf.length ? buf[buf.length >> 1] : 0
    }
    this.spreadCache.set(model, out)
    return out
  }

  /** Mean over the cells within `radius` meters of a point, for city badges. */
  areaMean(at: LngLat, lead: number, model: ModelId, radius = 900) {
    const [x, y] = utm(at)
    const c0 = Math.floor((x - GRID.x0 - radius) / GRID.cell)
    const c1 = Math.floor((x - GRID.x0 + radius) / GRID.cell)
    const r0 = Math.floor((GRID.y0 - y - radius) / GRID.cell)
    const r1 = Math.floor((GRID.y0 - y + radius) / GRID.cell)
    let s = 0
    let n = 0
    for (let row = Math.max(0, r0); row <= Math.min(GRID.rows - 1, r1); row += 2) {
      for (let col = Math.max(0, c0); col <= Math.min(GRID.cols - 1, c1); col += 2) {
        const v = this.value(row * GRID.cols + col, lead, model)
        if (!Number.isNaN(v)) {
          s += v
          n++
        }
      }
    }
    return n ? s / n : NaN
  }

  /** Synthetic feature attribution for one cell and hour, µg/m³ against a baseline. */
  explain(i: number, lead: number, model: ModelId) {
    const v = this.value(i, lead, model)
    const baseline = 19.5
    const delta = v - baseline
    const hour = localHourAt(this.clock, lead)
    const rush = clockBump(hour, 7.5, 1.5) + clockBump(hour, 19, 1.8)
    const rain = lead > this.clock.rainStart - 1
    const others = [
      { key: 'hour', label: 'Hour of day', w: rush > 0.4 ? 0.75 : -0.35 },
      { key: 'wind', label: 'Wind, east–west and north–south', w: -0.55 + 0.45 * gauss(lead, this.clock.episodePeak, 13) },
      { key: 'humidity', label: 'Relative humidity', w: 0.35 },
      { key: 'rain', label: 'Precipitation', w: rain ? -0.9 : -0.05 },
      { key: 'pressure', label: 'Surface pressure', w: 0.22 * gauss(lead, this.clock.episodePeak, 16) },
      { key: 'gusts', label: 'Wind gusts', w: -0.18 },
      { key: 'week', label: 'Day of week and month', w: 0.08 },
    ]
    // Weather and time terms get a fixed scale; recent history absorbs the rest,
    // so the parts always sum to the forecast.
    const scale = 2.4 * Math.max(0.6, v / baseline)
    const items = others.map((r) => ({ key: r.key, label: r.label, contribution: r.w * scale }))
    const rest = delta - items.reduce((s, r) => s + r.contribution, 0)
    items.push({ key: 'history', label: 'PM2.5 over the past 72 h', contribution: rest })
    items.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    return { baseline, value: v, items }
  }
}
