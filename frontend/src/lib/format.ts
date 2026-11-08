const TZ = 'Asia/Manila'

const timeFmt = new Intl.DateTimeFormat('en-PH', { timeZone: TZ, hour: 'numeric', minute: '2-digit' })
const hourFmt = new Intl.DateTimeFormat('en-PH', { timeZone: TZ, hour: 'numeric' })
const dayFmt = new Intl.DateTimeFormat('en-PH', { timeZone: TZ, weekday: 'short' })
const dateFmt = new Intl.DateTimeFormat('en-PH', { timeZone: TZ, weekday: 'short', day: 'numeric', month: 'short' })
const longFmt = new Intl.DateTimeFormat('en-PH', { timeZone: TZ, weekday: 'long', day: 'numeric', month: 'long' })

export const fmtTime = (ms: number) => timeFmt.format(ms)
export const fmtHour = (ms: number) => hourFmt.format(ms).replace(' ', '').toLowerCase()
export const fmtDay = (ms: number) => dayFmt.format(ms)
export const fmtDate = (ms: number) => dateFmt.format(ms)
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
  return `${h} h ${m % 60} min ago`
}

export const fmtPm = (v: number) => (Number.isNaN(v) ? '–' : v < 10 ? v.toFixed(1) : Math.round(v).toString())

export const fmtCoord = (lng: number, lat: number) => `${lat.toFixed(4)}° N, ${lng.toFixed(4)}° E`
