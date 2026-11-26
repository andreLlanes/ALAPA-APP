import { useMemo, useState } from 'react'
import { AQI, categoryOf } from '../data/aqi'
import { HORIZON, localHourAt, type Clock } from '../data/forecast'
import { dayWord, fmtGap, fmtHour, fmtPm } from '../lib/format'

interface Props {
  series: { v: Float32Array; lo: Float32Array; hi: Float32Array }
  clock: Clock
  lead: number
  worst: [number, number]
  onLead: (lead: number) => void
  /** A station's readings, hour k = lead k - lookback (NaN for none), when this place is a station. */
  measured?: { values: Float32Array; lookback: number } | null
  /** Whether the shaded band is a calibrated likely range (demo data) or kriging's range alone. */
  likelyRange: boolean
}

const W = 360
const H = 168
const PAD = { l: 30, r: 8, t: 20, b: 26 }
/** Hours of readings shown before the bulletin hour. The model saw 72; a day keeps the forecast legible. */
const PAST_H = 24

export function OutlookChart({ series, clock, lead, worst, onLead, measured, likelyRange }: Props) {
  const [hover, setHover] = useState<number | null>(null)
  const past = measured ? Math.min(PAST_H, measured.lookback) : 0

  const geo = useMemo(() => {
    const obs = (h: number) => (measured ? (measured.values[h + measured.lookback] ?? NaN) : NaN)
    let max = 0
    for (let h = 0; h < HORIZON; h++) if (Number.isFinite(series.hi[h])) max = Math.max(max, series.hi[h])
    for (let h = -past; h < HORIZON; h++) if (Number.isFinite(obs(h))) max = Math.max(max, obs(h))
    const ceil = AQI.find((c) => c.hi >= max * 1.08)?.hi ?? 250
    const top = Math.max(45, ceil)
    const span = HORIZON - 1 + past
    const x = (h: number) => PAD.l + ((h + past) / span) * (W - PAD.l - PAD.r)
    const y = (v: number) => PAD.t + (1 - Math.min(v, top) / top) * (H - PAD.t - PAD.b)
    let line = ''
    let upper = ''
    let lower = ''
    for (let h = 0; h < HORIZON; h++) {
      line += `${h ? 'L' : 'M'}${x(h).toFixed(1)},${y(series.v[h]).toFixed(1)}`
      upper += `${h ? 'L' : 'M'}${x(h).toFixed(1)},${y(series.hi[h]).toFixed(1)}`
    }
    for (let h = HORIZON - 1; h >= 0; h--) lower += `L${x(h).toFixed(1)},${y(series.lo[h]).toFixed(1)}`
    // Readings break where an hour is missing; an hour with no neighbor gets a dot instead.
    let reading = ''
    const lone: { h: number; v: number }[] = []
    for (let h = -past; h < HORIZON; h++) {
      const v = obs(h)
      if (!Number.isFinite(v)) continue
      const before = Number.isFinite(obs(h - 1)) && h > -past
      reading += `${before ? 'L' : 'M'}${x(h).toFixed(1)},${y(v).toFixed(1)}`
      if (!before && !Number.isFinite(obs(h + 1))) lone.push({ h, v })
    }
    const midnights: number[] = []
    for (let h = 1 - past; h < HORIZON; h++) if (h !== 0 && localHourAt(clock, h) === 0) midnights.push(h)
    return { obs, top, x, y, line, band: `${upper}${lower}Z`, reading, lone, midnights }
  }, [series, clock, measured, past])

  const { obs, top, x, y } = geo
  const shown = hover ?? lead
  const t = clock.issued + shown * 3_600_000
  const forecastShown = shown >= 0
  const cat = categoryOf(series.v[Math.max(0, shown)])
  const obsShown = obs(shown)

  const pointerToLead = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * W
    return Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (HORIZON - 1 + past)) - past
  }

  return (
    <figure className="outlook">
      <figcaption className="outlook__readout" aria-live="polite">
        <span className="outlook__when">
          {shown === 0 ? 'Now' : `${dayWord(t, clock.issued)}, ${fmtHour(t)}`}
        </span>
        {forecastShown ? (
          <span className="outlook__val">
            <span className="swatch" style={{ background: cat.color }} aria-hidden />
            <b className="num">{fmtPm(series.v[shown])}</b> µg/m³
            <span className="outlook__range">
              {likelyRange ? 'likely ' : 'map range '}
              <span className="num">
                {fmtPm(series.lo[shown])}–{fmtPm(series.hi[shown])}
              </span>
            </span>
          </span>
        ) : (
          <span className="outlook__val muted">Before the forecast</span>
        )}
      </figcaption>
      {measured && (
        <p className="outlook__measured">
          {Number.isFinite(obsShown) ? (
            <>
              <span className="swatch" style={{ background: categoryOf(obsShown).color }} aria-hidden />
              Measured here <b className="num">{fmtPm(obsShown)}</b> µg/m³
              {forecastShown && Number.isFinite(series.v[shown]) && (
                <span className="muted">
                  {' '}· forecast {fmtGap(series.v[shown] - obsShown)}
                </span>
              )}
            </>
          ) : (
            <span className="muted">No reading at this station this hour</span>
          )}
        </p>
      )}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="outlook__svg"
        role="img"
        aria-label={`Hourly PM2.5 outlook for the next 72 hours with the ${likelyRange ? 'likely range' : "map's kriging uncertainty"} shaded${measured ? `, and the readings measured at this station, from ${past} hours before` : ''}`}
        onPointerMove={(e) => {
          const l = pointerToLead(e)
          setHover(l >= -past && l < HORIZON ? l : null)
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
        {past > 0 && (
          <g>
            <rect x={x(-past)} width={x(0) - x(-past)} y={PAD.t} height={H - PAD.t - PAD.b} className="outlook__past" />
            <text x={x(-past) + 4} y={PAD.t + 10} className="outlook__tick">
              Past {past} h
            </text>
            <text x={x(0) + 4} y={PAD.t + 10} className="outlook__tick">
              Forecast
            </text>
          </g>
        )}
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
        {geo.midnights.map((h) => (
          <g key={h}>
            <line x1={x(h)} x2={x(h)} y1={PAD.t} y2={H - PAD.b + 4} className="outlook__day" />
            <text x={x(h) + 4} y={H - 8} className="outlook__tick">
              {dayWord(clock.issued + h * 3_600_000, clock.issued)}
            </text>
          </g>
        ))}
        {past === 0 && (geo.midnights[0] ?? HORIZON) > 9 && (
          <text x={PAD.l} y={H - 8} className="outlook__tick">
            Now
          </text>
        )}
        <path d={geo.band} className="outlook__band" />
        <path d={geo.line} className="outlook__line" />
        {measured && (
          <g className="outlook__reading">
            <path d={geo.reading} />
            {geo.lone.map((p) => (
              <circle key={p.h} cx={x(p.h)} cy={y(p.v)} r={1.8} />
            ))}
          </g>
        )}
        <line x1={x(lead)} x2={x(lead)} y1={PAD.t} y2={H - PAD.b} className="outlook__cursor" />
        <circle cx={x(lead)} cy={y(series.v[lead])} r={4} className="outlook__dot" />
        {Number.isFinite(obs(lead)) && <circle cx={x(lead)} cy={y(obs(lead))} r={3.2} className="outlook__obs-dot" />}
        {hover !== null && hover !== lead && (
          <line x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={H - PAD.b} className="outlook__hover" />
        )}
      </svg>
      <p className="outlook__legend">
        <span>
          <span className="outlook__key" aria-hidden /> Forecast
        </span>
        {measured && (
          <span>
            <span className="outlook__key outlook__key--reading" aria-hidden /> Measured at this station
          </span>
        )}
        <span>
          <span className="outlook__key outlook__key--band" aria-hidden />{' '}
          {likelyRange ? 'Likely range' : 'Map uncertainty (kriging), not forecast error'}
        </span>
      </p>
    </figure>
  )
}
