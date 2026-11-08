// DENR Air Quality Index for PM2.5 (24-hour mean, µg/m³).
// Breakpoints follow DAO 2020-14 as understood at design time; verify against
// the order before release. The Emergency ceiling is only an interpolation cap.

export type AqiKey = 'good' | 'fair' | 'usg' | 'very' | 'acute' | 'emergency'

export interface AqiCategory {
  key: AqiKey
  level: number
  name: string
  short: string
  lo: number
  hi: number
  indexLo: number
  indexHi: number
  color: string
  /** Text color that reads on `color`. */
  ink: string
  advice: string
}

export const AQI: AqiCategory[] = [
  {
    key: 'good', level: 1, name: 'Good', short: 'Good',
    lo: 0, hi: 25, indexLo: 0, indexHi: 50,
    color: '#3fa45b', ink: '#0d1a12',
    advice: 'Air is fine for everyone. A good window to be outside.',
  },
  {
    key: 'fair', level: 2, name: 'Fair', short: 'Fair',
    lo: 25, hi: 35, indexLo: 51, indexHi: 100,
    color: '#f1c21b', ink: '#1f1a05',
    advice: 'Acceptable for most people. If you are unusually sensitive, keep long, hard exercise outdoors short.',
  },
  {
    key: 'usg', level: 3, name: 'Unhealthy for sensitive groups', short: 'Sensitive',
    lo: 35, hi: 45, indexLo: 101, indexHi: 150,
    color: '#ef8a22', ink: '#1f1204',
    advice: 'Children, older adults, and people with asthma or heart conditions should limit long or heavy activity outdoors.',
  },
  {
    key: 'very', level: 4, name: 'Very unhealthy', short: 'Very unhealthy',
    lo: 45, hi: 55, indexLo: 151, indexHi: 200,
    color: '#c7321f', ink: '#ffffff',
    advice: 'Sensitive groups should stay indoors. Everyone else should cut back on outdoor exertion and keep windows shut near busy roads.',
  },
  {
    key: 'acute', level: 5, name: 'Acutely unhealthy', short: 'Acutely unhealthy',
    lo: 55, hi: 90, indexLo: 201, indexHi: 300,
    color: '#86399a', ink: '#ffffff',
    advice: 'Sensitive groups should stay indoors. Everyone should avoid exertion outside; wear a well-fitted mask if you must go out.',
  },
  {
    key: 'emergency', level: 6, name: 'Emergency', short: 'Emergency',
    lo: 90, hi: 250, indexLo: 301, indexHi: 500,
    color: '#6e1a2a', ink: '#ffffff',
    advice: 'Stay indoors with windows closed and avoid all outdoor activity.',
  },
]

export function categoryOf(pm: number): AqiCategory {
  for (const c of AQI) if (pm <= c.hi) return c
  return AQI[AQI.length - 1]
}

export function categoryIndex(pm: number): number {
  for (let i = 0; i < AQI.length; i++) if (pm <= AQI[i].hi) return i
  return AQI.length - 1
}

/** DENR AQI index value, linearly interpolated inside the category. */
export function aqiValue(pm: number): number {
  const c = categoryOf(pm)
  const t = Math.min(1, Math.max(0, (pm - c.lo) / (c.hi - c.lo)))
  return Math.round(c.indexLo + t * (c.indexHi - c.indexLo))
}

/** RGB triplets for the canvas painter, indexed like AQI. */
export const AQI_RGB: [number, number, number][] = AQI.map((c) => {
  const n = parseInt(c.color.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
})
