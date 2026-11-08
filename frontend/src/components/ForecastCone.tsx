import { useEffect, useMemo, useRef, useState } from 'react'
import { AQI } from '../data/aqi'
import { HORIZON, localHourAt, type Clock } from '../data/forecast'
import { dayWord, fmtHour } from '../lib/format'

interface Props {
  share: Float32Array
  /** Median interval half-width per hour, µg/m³; sets how wide the cone opens. */
  spread: Float32Array
  clock: Clock
  lead: number
  onLead: (lead: number) => void
}

const AXIS = 22
const PAD_X = 16

/**
 * The Metro outlook cone: each hour's column shows the share of Metro Manila in
 * each AQI category, inside an envelope that widens as forecast uncertainty grows.
 * The whole strip is the hour slider.
 */
export function ForecastCone({ share, spread, clock, lead, onLead }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(900)
  const dragging = useRef(false)

  useEffect(() => {
    if (!box.current) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(320, e.contentRect.width)))
    ro.observe(box.current)
    return () => ro.disconnect()
  }, [])

  const H = width < 560 ? 84 : 112
  const geo = useMemo(() => {
    const inner = width - PAD_X * 2
    const col = inner / HORIZON
    const mid = (H - AXIS) / 2 + 2
    // The envelope reports the forecast's own uncertainty: each hour's half-height
    // is its median interval half-width, scaled so the widest hour fills the band.
    const narrow = (H - AXIS) * 0.2
    const wide = (H - AXIS) / 2 - 4
    let lo = Infinity
    let hi = 0
    for (const v of spread) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    const span = hi - lo
    const half = (h: number) =>
      span > 1e-9 ? narrow + ((spread[h] - lo) / span) * (wide - narrow) : (narrow + wide) / 2
    const x = (h: number) => PAD_X + h * col
    const days: { start: number; end: number; label: string; peak: number }[] = []
    let start = 0
    for (let h = 1; h <= HORIZON; h++) {
      if (h === HORIZON || localHourAt(clock, h) === 0) {
        let peak = start
        let peakScore = -1
        for (let k = start; k < h; k++) {
          let score = 0
          for (let c = 0; c < AQI.length; c++) score += share[k * AQI.length + c] * c
          if (score > peakScore) {
            peakScore = score
            peak = k
          }
        }
        days.push({ start, end: h, label: dayWord(clock.issued + start * 3_600_000, clock.issued), peak })
        start = h
      }
    }
    let outline = ''
    for (let h = 0; h < HORIZON; h++) outline += `${h ? 'L' : 'M'}${x(h).toFixed(1)},${(mid - half(h)).toFixed(1)}`
    outline += `L${(x(HORIZON - 1) + col).toFixed(1)},${(mid - half(HORIZON - 1)).toFixed(1)}`
    outline += `L${(x(HORIZON - 1) + col).toFixed(1)},${(mid + half(HORIZON - 1)).toFixed(1)}`
    for (let h = HORIZON - 1; h >= 0; h--) outline += `L${x(h).toFixed(1)},${(mid + half(h)).toFixed(1)}`
    outline += 'Z'
    return { col, mid, half, x, days, outline }
  }, [width, share, spread, clock, H])

  const leadFromX = (clientX: number) => {
    const r = box.current!.getBoundingClientRect()
    return Math.max(0, Math.min(HORIZON - 1, Math.floor((clientX - r.left - PAD_X) / geo.col)))
  }

  const t = clock.issued + lead * 3_600_000
  const when = lead === 0 ? `Now, ${fmtHour(t)}` : `${dayWord(t, clock.issued)}, ${fmtHour(t)}`

  const onKey = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 6 : 1
    const map: Record<string, number> = {
      ArrowRight: lead + step,
      ArrowUp: lead + step,
      ArrowLeft: lead - step,
      ArrowDown: lead - step,
      PageUp: lead + 24,
      PageDown: lead - 24,
      Home: 0,
      End: HORIZON - 1,
    }
    if (e.key in map) {
      e.preventDefault()
      onLead(Math.max(0, Math.min(HORIZON - 1, map[e.key])))
    }
  }

  const cursorX = geo.x(lead) + geo.col / 2

  return (
    <section className="cone" aria-label="72-hour outlook for the coverage area">
      <header className="cone__head">
        <h2 className="cone__title">
          Area outlook<span className="cone__title-long">, next 72 hours</span>
        </h2>
        <p className="cone__note">
          Share of the coverage area at each level. The band is tallest where the
          forecast range is widest.
        </p>
        <nav className="cone__days" aria-label="Jump to a day's worst hour">
          {geo.days
            .filter((d) => d.end - d.start >= 3)
            .map((d) => (
              <button
                key={d.start}
                type="button"
                className="cone__day"
                aria-pressed={lead >= d.start && lead < d.end}
                onClick={() => onLead(d.peak)}
              >
                {d.label}
                <span className="cone__day-sub">worst {fmtHour(clock.issued + d.peak * 3_600_000)}</span>
              </button>
            ))}
        </nav>
      </header>
      <div
        ref={box}
        style={{ height: H }}
        className="cone__track"
        role="slider"
        tabIndex={0}
        aria-label="Forecast hour"
        aria-valuemin={0}
        aria-valuemax={HORIZON - 1}
        aria-valuenow={lead}
        aria-valuetext={when}
        onKeyDown={onKey}
        onPointerDown={(e) => {
          dragging.current = true
          e.currentTarget.setPointerCapture(e.pointerId)
          onLead(leadFromX(e.clientX))
        }}
        onPointerMove={(e) => {
          if (dragging.current) onLead(leadFromX(e.clientX))
        }}
        onPointerUp={() => {
          dragging.current = false
        }}
      >
        <svg width={width} height={H} className="cone__svg" aria-hidden>
          <defs>
            <clipPath id="cone-clip">
              <path d={geo.outline} />
            </clipPath>
          </defs>
          <g clipPath="url(#cone-clip)">
            {Array.from({ length: HORIZON }, (_, h) => {
              const top = geo.mid - geo.half(h)
              const span = geo.half(h) * 2
              let acc = 0
              return AQI.map((c, k) => {
                const s = share[h * AQI.length + k]
                if (s <= 0) return null
                const hgt = s * span
                // Worst levels sit on top of the column.
                const yy = top + span - acc - hgt
                acc += hgt
                return <rect key={`${h}-${c.key}`} x={geo.x(h)} y={yy} width={geo.col + 0.6} height={hgt} fill={c.color} />
              })
            })}
          </g>
          <path d={geo.outline} className="cone__outline" />
          {geo.days.slice(1).map((d) => (
            <line key={d.start} x1={geo.x(d.start)} x2={geo.x(d.start)} y1={4} y2={H - AXIS + 6} className="cone__midnight" />
          ))}
          {Array.from({ length: HORIZON }, (_, h) => h)
            .filter((h) => localHourAt(clock, h) % (width < 560 ? 12 : 6) === 0)
            .map((h) => (
              <g key={h}>
                <line x1={geo.x(h) + geo.col / 2} x2={geo.x(h) + geo.col / 2} y1={H - AXIS + 2}
                      y2={H - AXIS + 6} className="cone__tick" />
                <text x={Math.max(PAD_X + 2, geo.x(h) + geo.col / 2)} y={H - 6} className="cone__label"
                      textAnchor={geo.x(h) + geo.col / 2 < PAD_X + 12 ? 'start' : 'middle'}>
                  {fmtHour(clock.issued + h * 3_600_000)}
                </text>
              </g>
            ))}
          <line x1={cursorX} x2={cursorX} y1={0} y2={H - AXIS + 6} className="cone__cursor" />
        </svg>
        <div
          className="cone__handle"
          style={{ left: Math.max(64, Math.min(width - 64, cursorX)) }}
          aria-hidden
        >
          {when}
        </div>
      </div>
    </section>
  )
}
