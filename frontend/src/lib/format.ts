const TZ = 'Asia/Manila'

const timeFmt = new Intl.DateTimeFormat('en-PH', { timeZone: TZ, hour: 'numeric', minute: '2-digit' })
const hourFmt = new Intl.DateTimeFormat('en-PH', { timeZone: TZ, hour: 'numeric' })
const dayFmt = new Intl.DateTimeFormat('en-PH', { timeZone: TZ, weekday: 'short' })
const dateFmt = new Intl.DateTimeFormat('en-PH', { timeZone: TZ, weekday: 'short', day: 'numeric', month: 'short' })
const yearFmt = new Intl.DateTimeFormat('en-PH', { timeZone: TZ, year: 'numeric' })
const longFmt =new Intl.DateTimeFormat('en-PH', { timeZone: TZ, weekday: 'long', day: 'numeric', month: 'long' })

export const fmtTime = (ms: number) => timeFmt.format(ms)
export const fmtHour = (ms: number) => hourFmt.format(ms).replace(' ', '').toLowerCase()
export const fmtDay = (ms: number) => dayFmt.format(ms)
export const fmtDate = (ms: number) => dateFmt.format(ms)
export const fmtYear = (ms: number) => yearFmt.format(ms)
export const fmtLongDate = (ms: number) => longFmt.format(ms)

/** "Today", "Tomorrow", or the weekday, relative to `ref`. */
export function dayWord(ms: number, ref: number) {
  const key = (t: number) => new Intl.DateTimeFormat('en-CA', { timeZone: TZ }).format(t)
  const d = Math.round((Date.parse(key(ms)) - Date.parse(key(ref))) / 86_400_000)
  if (d === 0) return 'Today'
  if (d === 1) return 'Tomorrow'
  return longFmt.format(ms).split(',')[0]
}

export function fmtAgo(ms: number, now: number) {
  const m = Math.max(0, Math.round((now - ms) / 60_000))
  if (m < 1) return 'just now'
  if (m < 60) return `${m} min ago`
  const h = Math.floor(m / 60)
  if (h < 48) return `${h} h ${m % 60} min ago`
  return `${Math.floor(h / 24)} days ago`
}

export const fmtPm = (v: number) => (Number.isNaN(v) ? '–' : v < 10 ? v.toFixed(1) : Math.round(v).toString())

/** A forecast-minus-reading gap in words: "5.2 higher", "3.1 lower", or "on target". */
export function fmtGap(d: number) {
  if (Math.abs(d) < 0.5) return 'on target'
  return `${fmtPm(Math.abs(d))} ${d > 0 ? 'higher' : 'lower'}`
}

/** A signed µg/m³ difference: "+5.2", "−3.1". */
export const fmtSigned = (d: number) => (Number.isNaN(d) ? '–' : `${d >= 0 ? '+' : '−'}${fmtPm(Math.abs(d))}`)

export const fmtCoord = (lng: number, lat: number) => `${lat.toFixed(4)}° N, ${lng.toFixed(4)}° E`
