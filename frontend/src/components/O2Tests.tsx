import { X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { O2_MODELS, type O2Model, type O2Tests as Data } from '../data/o2Tests'

// O2's error progression in its own dialog: how far ahead the station forecasts hold up on
// O2's Metro Manila test period. It sits apart from the map's check on purpose: O2 and O3 use
// different splits and stations, so their numbers are never compared.

interface Props {
  data: Data
  /** The app's model, for the tab the dialog opens on. */
  model: string
  /** The app's hour (0 = the first forecast hour), marked on the chart. */
  lead: number
  onClose: () => void
}

const NAMES: Record<O2Model, string> = { lstm: 'LSTM', gnn: 'GNN', gbt: 'GBT' }
const SHORT: Record<string, string> = { mm: 'Manila only', bk: 'Bangkok', la: 'Los Angeles' }
const COLOR: Record<string, string> = { mm: 'var(--series-1)', bk: 'var(--series-2)', la: 'var(--series-3)' }
const BASE = [
  { id: 'persistence', label: 'Persistence', dash: '5 4' },
  { id: 'climatology', label: 'Climatology', dash: '1.5 3' },
] as const

/** Drawn at the figure's own width, so the labels stay legible on a phone. */
const MIN_W = 300
const PAD = { l: 40, r: 96, t: 14, b: 34 }
const LEADS = 72
const LABEL_GAP = 11

const dayFmt = new Intl.DateTimeFormat('en-PH', { timeZone: 'Asia/Manila', day: 'numeric', month: 'short', year: 'numeric' })
const one = (v: number) => v.toFixed(1)

interface Series {
  id: string
  label: string
  values: number[]
  color: string
  dash?: string
  baseline: boolean
}

/** Spread end labels so none overlap: keep order, push apart by LABEL_GAP, stay inside the plot. */
function spread(ys: number[], top: number, bottom: number): number[] {
  const order = ys.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y)
  const out = order.map((o) => o.y)
  for (let k = 1; k < out.length; k++) out[k] = Math.max(out[k], out[k - 1] + LABEL_GAP)
  const over = out[out.length - 1] - bottom
  if (over > 0) for (let k = 0; k < out.length; k++) out[k] -= over
  for (let k = 0; k < out.length; k++) out[k] = Math.max(out[k], top + k * LABEL_GAP)
  const res = new Array<number>(ys.length)
  order.forEach((o, k) => (res[o.i] = out[k]))
  return res
}

export function O2Tests({ data, model, lead, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [tab, setTab] = useState<O2Model>(() => (O2_MODELS as readonly string[]).includes(model) ? (model as O2Model) : 'gbt')
  const [hover, setHover] = useState<number | null>(null)
  const [table, setTable] = useState(false)
  const figure = useRef<HTMLElement>(null)
  const [W, setW] = useState(640)
  const H = W < 520 ? 250 : 300

  useEffect(() => {
    const d = dialog.current
    if (!d || d.open) return
    d.showModal()
    // Start on the model tabs, the first thing to change, rather than on the close button.
    d.querySelector<HTMLInputElement>('input[name=o2-model]:checked')?.focus()
  }, [])

  useEffect(() => {
    const el = figure.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(MIN_W, Math.round(e.contentRect.width))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [table])

  const series: Series[] = useMemo(
    () => [
      ...data.models[tab].lines.map((l) => ({
        id: l.id,
        label: SHORT[l.id] ?? l.label,
        values: l.rmse,
        color: COLOR[l.id] ?? 'var(--ink)',
        baseline: false,
      })),
      ...BASE.map((b) => ({
        id: b.id,
        label: b.label,
        values: data.baselines[b.id],
        color: 'var(--ink-3)',
        dash: b.dash,
        baseline: true,
      })),
    ],
    [data, tab],
  )

  const geo = useMemo(() => {
    let max = 0
    for (const s of series) for (const v of s.values) max = Math.max(max, v)
    const top = Math.ceil((max * 1.05) / 5) * 5
    const x = (h: number) => PAD.l + ((h - 1) / (LEADS - 1)) * (W - PAD.l - PAD.r)
    const y = (v: number) => PAD.t + (1 - v / top) * (H - PAD.t - PAD.b)
    const paths = series.map((s) => s.values.map((v, i) => `${i ? 'L' : 'M'}${x(i + 1).toFixed(1)},${y(v).toFixed(1)}`).join(''))
    const ends = spread(series.map((s) => y(s.values[LEADS - 1])), PAD.t + 4, H - PAD.b - 2)
    const yTicks: number[] = []
    for (let v = 0; v <= top; v += 5) yTicks.push(v)
    return { top, x, y, paths, ends, yTicks }
  }, [series, W, H])

  const { x, y } = geo
  const viewed = Math.min(LEADS, Math.max(1, lead + 1))
  const shown = hover ?? viewed
  const lines = data.models[tab].lines
  const from = dayFmt.format(Date.parse(data.test.first_origin_utc))
  const to = dayFmt.format(Date.parse(data.test.last_origin_utc))

  const pointerToHour = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * W
    return Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (LEADS - 1)) + 1
  }

  return (
    <dialog ref={dialog} className="tests" aria-labelledby="tests-title" onClose={onClose}>
      <header className="tests__head">
        <div>
          <h2 id="tests-title" className="tests__title">
            How far ahead the forecasts hold up
          </h2>
          <p className="tests__sub">
            O2&rsquo;s station-forecast tests on Metro Manila, forecasts made {from} to {to}, scored at{' '}
            <span className="num">{data.test.stations_scored}</span> stations. Each line is the mean of{' '}
            <span className="num">{data.seeds}</span> trained copies.
          </p>
        </div>
        <button type="button" className="tests__close" aria-label="Close" onClick={() => dialog.current?.close()}>
          <X size={18} aria-hidden />
        </button>
      </header>

      <div className="tests__bar">
        <div className="seg" role="radiogroup" aria-label="Model">
          {O2_MODELS.map((m) => (
            <label key={m} className="seg__opt">
              <input type="radio" name="o2-model" value={m} checked={tab === m} onChange={() => setTab(m)} />
              <span>{NAMES[m]}</span>
            </label>
          ))}
        </div>
        <button type="button" className="linkbtn tests__toggle" onClick={() => setTable(!table)} aria-pressed={table}>
          {table ? 'Show as chart' : 'Show as table'}
        </button>
      </div>

      <ul className="tests__legend">
        {lines.map((l) => (
          <li key={l.id}>
            <span className="tests__key" style={{ borderColor: COLOR[l.id] }} aria-hidden />
            <b>{l.label}</b>
            {l.variant !== 'none' && <span className="muted"> · {l.how}</span>}
          </li>
        ))}
        {BASE.map((b) => (
          <li key={b.id}>
            <span className={`tests__key tests__key--${b.id}`} aria-hidden />
            {b.label}
          </li>
        ))}
      </ul>

      {table ? (
        <div className="tests__scroll">
          <table className="tests__table">
            <caption className="sr-only">RMSE in µg/m³ at each hour ahead, {NAMES[tab]}</caption>
            <thead>
              <tr>
                <th scope="col">Hours ahead</th>
                {series.map((s) => (
                  <th key={s.id} scope="col">
                    {s.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: LEADS }, (_, i) => (
                <tr key={i} className={i + 1 === viewed ? 'is-viewed' : undefined}>
                  <th scope="row" className="num">
                    {i + 1}
                  </th>
                  {series.map((s) => (
                    <td key={s.id} className="num">
                      {one(s.values[i])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <figure className="tests__figure" ref={figure}>
          <figcaption className="tests__readout" aria-live="polite">
            <span className="tests__when">
              <span className="num">{shown}</span> h ahead{shown === viewed && hover === null ? ', the hour you are viewing' : ''}
            </span>
            {series.map((s) => (
              <span key={s.id} className="tests__val">
                {s.label} <b className="num">{one(s.values[shown - 1])}</b>
              </span>
            ))}
          </figcaption>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="tests__svg"
            role="img"
            aria-label={`${NAMES[tab]} test RMSE by hours ahead, 1 to 72, for ${lines.map((l) => l.label).join(', ')}, with persistence and climatology`}
            onPointerMove={(e) => {
              const h = pointerToHour(e)
              setHover(h >= 1 && h <= LEADS ? h : null)
            }}
            onPointerLeave={() => setHover(null)}
          >
            {geo.yTicks.map((v) => (
              <g key={v}>
                <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className={v === 0 ? 'tests__axis' : 'tests__grid'} />
                <text x={PAD.l - 6} y={y(v) + 3} className="tests__tick" textAnchor="end">
                  {v}
                </text>
              </g>
            ))}
            {[1, 12, 24, 36, 48, 60, 72].map((h) => (
              <text key={h} x={x(h)} y={H - PAD.b + 14} className="tests__tick" textAnchor="middle">
                {h}
              </text>
            ))}
            <text x={(PAD.l + W - PAD.r) / 2} y={H - 4} className="tests__axis-title" textAnchor="middle">
              Hours ahead
            </text>
            <text x={12} y={PAD.t + (H - PAD.t - PAD.b) / 2} className="tests__axis-title" textAnchor="middle"
                  transform={`rotate(-90 12 ${PAD.t + (H - PAD.t - PAD.b) / 2})`}>
              RMSE (µg/m³)
            </text>

            <line x1={x(viewed)} x2={x(viewed)} y1={PAD.t} y2={H - PAD.b} className="tests__marker" />
            {hover !== null && hover !== viewed && (
              <line x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={H - PAD.b} className="tests__hover" />
            )}

            {series.map((s, i) => (
              <path
                key={s.id}
                d={geo.paths[i]}
                className={s.baseline ? 'tests__base' : 'tests__line'}
                style={{ stroke: s.color }}
                strokeDasharray={s.dash}
              />
            ))}
            {series.map((s) => (
              <circle key={s.id} cx={x(shown)} cy={y(s.values[shown - 1])} r={3.5} className="tests__dot"
                      style={{ stroke: s.color }} />
            ))}
            {series.map((s, i) => (
              <text key={s.id} x={W - PAD.r + 8} y={geo.ends[i] + 3.5} className="tests__end">
                {s.label}
              </text>
            ))}
          </svg>
        </figure>
      )}

      <p className="footnote tests__note">
        RMSE is the typical miss at the stations in µg/m³, so lower is better. These are O2&rsquo;s tests of the station
        forecasts (split {data.settings.split}, stations at least {data.settings.completeness}% complete). The map comes from
        O3&rsquo;s Manila-only models, trained and tested separately; its check is the &ldquo;All 72 h&rdquo; table in the
        coverage-area panel. The two use
        different splits and stations, so their numbers are not comparable.
      </p>
    </dialog>
  )
}
