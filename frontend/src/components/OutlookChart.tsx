import { useMemo, useState } from 'react'
import { AQI, categoryOf } from '../data/aqi'
import { HORIZON, localHourAt, type Clock } from '../data/forecast'
import { dayWord, fmtHour, fmtPm } from '../lib/format'

interface Props {
  series: { v: Float32Array; lo: Float32Array; hi: Float32Array }
  clock: Clock
  lead: number
  worst: [number, number]
  onLead: (lead: number) => void
}

const W = 360
const H = 168
const PAD = { l: 30, r: 8, t: 20, b: 26 }

export function OutlookChart({ series, clock, lead, worst, onLead }: Props) {
  const [hover, setHover] = useState<number | null>(null)

  const { top, x, y, line, band, midnights } = useMemo(() => {
    let max = 0
    for (let h = 0; h < HORIZON; h++) max = Math.max(max, series.hi[h])
    const ceil = AQI.find((c) => c.hi >= max * 1.08)?.hi ?? 250
    const top = Math.max(45, ceil)
    const x = (h: number) => PAD.l + (h / (HORIZON - 1)) * (W - PAD.l - PAD.r)
    const y = (v: number) => PAD.t + (1 - Math.min(v, top) / top) * (H - PAD.t - PAD.b)
    let line = ''
    let upper = ''
    let lower = ''
    for (let h = 0; h < HORIZON; h++) {
      line += `${h ? 'L' : 'M'}${x(h).toFixed(1)},${y(series.v[h]).toFixed(1)}`
      upper += `${h ? 'L' : 'M'}${x(h).toFixed(1)},${y(series.hi[h]).toFixed(1)}`
    }
    for (let h = HORIZON - 1; h >= 0; h--) lower += `L${x(h).toFixed(1)},${y(series.lo[h]).toFixed(1)}`
    const midnights: number[] = []
    for (let h = 1; h < HORIZON; h++) if (localHourAt(clock, h) === 0) midnights.push(h)
    return { top, x, y, line, band: `${upper}${lower}Z`, midnights }
  }, [series, clock])

  const shown = hover ?? lead
  const t = clock.issued + shown * 3_600_000
  const cat = categoryOf(series.v[shown])

  const pointerToLead = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * W
    return Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (HORIZON - 1))
  }

  return (
    <figure className="outlook">
      <figcaption className="outlook__readout" aria-live="polite">
        <span className="outlook__when">
          {shown === 0 ? 'Now' : `${dayWord(t, clock.issued)}, ${fmtHour(t)}`}
        </span>
        <span className="outlook__val">
          <span className="swatch" style={{ background: cat.color }} aria-hidden />
          <b className="num">{fmtPm(series.v[shown])}</b> µg/m³
          <span className="outlook__range num">
            {fmtPm(series.lo[shown])}–{fmtPm(series.hi[shown])}
          </span>
        </span>
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="outlook__svg"
        role="img"
        aria-label="Hourly PM2.5 outlook for the next 72 hours with the likely range shaded"
        onPointerMove={(e) => {
          const l = pointerToLead(e)
          setHover(l >= 0 && l < HORIZON ? l : null)
        }}
        onPointerLeave={() => setHover(null)}
        onClick={(e) => {
          const l = pointerToLead(e)
          if (l >= 0 && l < HORIZON) onLead(l)
        }}
      >
        {AQI.filter((c) => c.lo < top).map((c) => (
          <rect
            key={c.key}
            x={PAD.l}
            width={W - PAD.l - PAD.r}
            y={y(Math.min(c.hi, top))}
            height={y(c.lo) - y(Math.min(c.hi, top))}
            fill={c.color}
            opacity={0.16}
          />
        ))}
        {AQI.filter((c) => c.hi < top).map((c) => (
          <g key={c.key}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(c.hi)} y2={y(c.hi)} className="outlook__rule" />
            <text x={PAD.l - 5} y={y(c.hi) + 3} className="outlook__tick" textAnchor="end">
              {c.hi}
            </text>
          </g>
        ))}
        <g>
          <rect
            x={x(worst[0]) - 2}
            width={x(worst[1]) - x(worst[0]) + 4}
            y={PAD.t}
            height={H - PAD.t - PAD.b}
            className="outlook__worst"
          />
          <text x={(x(worst[0]) + x(worst[1])) / 2} y={PAD.t - 1} className="outlook__worst-label"
                textAnchor="middle">
            worst
          </text>
        </g>
        {midnights.map((h) => (
          <g key={h}>
            <line x1={x(h)} x2={x(h)} y1={PAD.t} y2={H - PAD.b + 4} className="outlook__day" />
            <text x={x(h) + 4} y={H - 8} className="outlook__tick">
              {dayWord(clock.issued + h * 3_600_000, clock.issued)}
            </text>
          </g>
        ))}
        {(midnights[0] ?? HORIZON) > 9 && (
          <text x={PAD.l} y={H - 8} className="outlook__tick">
            Now
          </text>
        )}
        <path d={band} className="outlook__band" />
        <path d={line} className="outlook__line" />
        <line x1={x(lead)} x2={x(lead)} y1={PAD.t} y2={H - PAD.b} className="outlook__cursor" />
        <circle cx={x(lead)} cy={y(series.v[lead])} r={4} className="outlook__dot" />
        {hover !== null && hover !== lead && (
          <line x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={H - PAD.b} className="outlook__hover" />
        )}
      </svg>
    </figure>
  )
}
