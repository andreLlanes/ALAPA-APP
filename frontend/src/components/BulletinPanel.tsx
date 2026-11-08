import { ChevronDown, LocateFixed, Minus, TrendingDown, TrendingUp } from 'lucide-react'
import { useMemo, useState } from 'react'
import { AQI, aqiValue, categoryOf } from '../data/aqi'
import { HORIZON, MODELS, localHourAt, type Clock, type ForecastField, type ModelId } from '../data/forecast'
import { cellCenter, metersBetween, type CellRef } from '../data/grid'
import { CITIES } from '../data/places'
import { dayWord, fmtCoord, fmtHour, fmtPm } from '../lib/format'
import { Explain } from './Explain'
import { OutlookChart } from './OutlookChart'

export type LocStatus = 'asking' | 'granted' | 'denied' | 'unsupported' | 'outside' | 'picked'

interface Props {
  field: ForecastField
  clock: Clock
  cell: CellRef | null
  placeName: string
  locStatus: LocStatus
  lead: number
  model: ModelId
  onLead: (lead: number) => void
  onModel: (m: ModelId) => void
  onLocate: () => void
  onCity: (id: string) => void
}

const TREND = {
  rising: { icon: TrendingUp, text: 'Rising' },
  easing: { icon: TrendingDown, text: 'Easing' },
  steady: { icon: Minus, text: 'Steady' },
}

export function BulletinPanel(props: Props) {
  const { cell } = props
  return (
    <aside className="panel" aria-label="Bulletin for the selected place">
      {cell ? <Bulletin {...props} cell={cell} /> : <NoPlace {...props} />}
    </aside>
  )
}

function NoPlace({ field, clock, lead, model, locStatus, onLocate, onCity, title, lede }: Props & { title?: string; lede?: string }) {
  const message: Record<LocStatus, string> = {
    asking: 'Finding your location…',
    granted: 'Finding your location…',
    picked: '',
    denied: 'Location is off, so this is the bulletin for the whole coverage area: Metro Manila and the edges of nearby provinces inside the dashed line.',
    unsupported: 'This browser cannot share a location.',
    outside: "You appear to be outside ALAPA's coverage area, the Metro Manila study grid.",
  }
  const t = clock.issued + lead * 3_600_000
  const when = lead === 0 ? 'now' : `${dayWord(t, clock.issued).toLowerCase()}, ${fmtHour(t)}`
  const groups = useMemo(() => {
    const rows = CITIES.map((c) => ({ city: c, v: field.areaMean(c.at, lead, model) }))
    return AQI.map((cat) => ({
      cat,
      rows: rows.filter((r) => categoryOf(r.v).key === cat.key).sort((a, b) => b.v - a.v),
    }))
      .filter((g) => g.rows.length)
      .reverse()
  }, [field, lead, model])

  return (
    <div className="panel__empty">
      <h2 className="panel__title">{title ?? `Coverage area, ${when}`}</h2>
      <p className="panel__lede">
        {lede ?? `${message[locStatus]} Tap the map or a city for its 72-hour bulletin.`}
      </p>
      <ol className="areas" aria-label="Metro Manila cities by level">
        {groups.map((g) => (
          <li key={g.cat.key} className="areas__group" style={{ '--cat': g.cat.color, '--cat-ink': g.cat.ink } as React.CSSProperties}>
            <h3 className="areas__level">
              <span className="areas__no">{g.cat.level}</span>
              {g.cat.name}
            </h3>
            <ul className="areas__list">
              {g.rows.map((r) => (
                <li key={r.city.id}>
                  <button type="button" className="areas__city" onClick={() => onCity(r.city.id)}>
                    <span>{r.city.name}</span>
                    <span className="num muted">{fmtPm(r.v)}</span>
                  </button>
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
      <p className="footnote">The 17 Metro Manila cities, by the average around each city center (µg/m³ PM2.5).</p>
      <label className="field">
        <span className="field__label">Go to a city</span>
        <select className="select" defaultValue="" onChange={(e) => e.target.value && onCity(e.target.value)}>
          <option value="" disabled>
            Choose a city
          </option>
          {[...CITIES].sort((a, b) => a.name.localeCompare(b.name)).map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
      </label>
      {locStatus !== 'unsupported' && locStatus !== 'asking' && (
        <button type="button" className="btn btn--ghost" onClick={onLocate}>
          <LocateFixed size={16} aria-hidden /> Use my location
        </button>
      )}
    </div>
  )
}

function Bulletin({ field, clock, cell, placeName, locStatus, lead, model, onLead, onModel, onLocate, onCity }: Props & { cell: CellRef }) {
  const [open, setOpen] = useState(() => new URLSearchParams(window.location.search).has('details'))
  const series = useMemo(() => field.series(cell.index, model), [field, cell.index, model])
  const water = field.isWater(cell.index)

  const worst = useMemo((): [number, number] => {
    let peak = 0
    for (let h = 1; h < HORIZON; h++) if (series.v[h] > series.v[peak]) peak = h
    const cat = categoryOf(series.v[peak]).key
    let a = peak
    let b = peak
    while (a > 0 && categoryOf(series.v[a - 1]).key === cat && series.v[a - 1] >= series.v[peak] * 0.85) a--
    while (b < HORIZON - 1 && categoryOf(series.v[b + 1]).key === cat && series.v[b + 1] >= series.v[peak] * 0.85) b++
    return [a, b]
  }, [series])

  const days = useMemo(() => {
    const out: { label: string; mean: number; hours: number }[] = []
    let sum = 0
    let n = 0
    let start = 0
    for (let h = 0; h <= HORIZON; h++) {
      if (h === HORIZON || (h > 0 && localHourAt(clock, h) === 0)) {
        if (n >= 6) {
          const t = clock.issued + start * 3_600_000
          out.push({ label: start === 0 ? 'Rest of today' : dayWord(t, clock.issued), mean: sum / n, hours: n })
        }
        sum = 0
        n = 0
        start = h
      }
      if (h < HORIZON) {
        sum += series.v[h]
        n++
      }
    }
    return out.slice(0, 3)
  }, [series, clock])

  if (water) {
    return (
      <NoPlace field={field} clock={clock} cell={null} placeName="" locStatus="picked" lead={lead} model={model}
        onLead={onLead} onModel={onModel} onLocate={onLocate} onCity={onCity}
        title="That spot is water"
        lede="ALAPA estimates PM2.5 over land only. Tap a spot on land, or choose a city." />
    )
  }

  const v = series.v[lead]
  const cat = categoryOf(v)
  const trend = TREND[field.trend(cell.index, lead, model)]
  const TrendIcon = trend.icon
  const t = clock.issued + lead * 3_600_000
  const peakT = clock.issued + worst[0] * 3_600_000
  const peakEndT = clock.issued + (worst[1] + 1) * 3_600_000
  const peakCat = categoryOf(Math.max(...series.v.slice(worst[0], worst[1] + 1)))
  const dist = field.stationDistance[cell.index]
  const conf = field.confidence(cell.index)
  const confWord = conf > 0.6 ? 'Higher' : conf > 0.3 ? 'Moderate' : 'Lower'
  const [lng, lat] = cellCenter(cell)
  const city = CITIES.find((c) => metersBetween(c.at, [lng, lat]) < 1500)
  const cityMean = city ? field.areaMean(city.at, lead, model) : NaN
  const cityCat = categoryOf(cityMean)

  return (
    <div className="bulletin" style={{ '--cat': cat.color, '--cat-ink': cat.ink } as React.CSSProperties}>
      <header className="bulletin__place">
        <h2 className="panel__title">
          {locStatus === 'granted' && <LocateFixed size={20} aria-label="Your location" className="bulletin__here" />}
          {placeName}
        </h2>
      </header>

      <section className="level" aria-live="polite">
        <div className="level__band">
          <span className="level__when">{lead === 0 ? 'Now' : `${dayWord(t, clock.issued)}, ${fmtHour(t)}`}</span>
          <span className="level__name">{cat.name}</span>
        </div>
        <div className="level__body">
          <p className="level__reading">
            <span className="level__pm num">{fmtPm(v)}</span>
            <span className="level__unit">µg/m³ PM2.5</span>
          </p>
          <p className="level__trend">
            <TrendIcon size={16} aria-hidden /> {trend.text} <span className="muted">vs. 3 h earlier</span>
          </p>
          <p className="level__range">
            Likely between <b className="num">{fmtPm(series.lo[lead])}</b> and <b className="num">{fmtPm(series.hi[lead])}</b>
          </p>
        </div>
      </section>

      <p className="advice">{cat.advice}</p>
      {city && (
        <p className="area-note">
          <span className="swatch" style={{ background: cityCat.color }} aria-hidden />
          <span>
            Around central {city.name}, the area average is <span className="num">{fmtPm(cityMean)}</span> µg/m³ ({cityCat.name.toLowerCase()}).
            This spot is one 100 m cell.
          </span>
        </p>
      )}

      <section className="worst">
        <div className="worst__row">
          <span className="swatch swatch--lg" style={{ background: peakCat.color }} aria-hidden />
          <div>
            <h3 className="worst__when">
              <button type="button" className="worst__link" onClick={() => onLead(worst[0])}>
                Worst ahead: {dayWord(peakT, clock.issued)}, {fmtHour(peakT)} to{' '}
                {dayWord(peakEndT, clock.issued) === dayWord(peakT, clock.issued)
                  ? fmtHour(peakEndT)
                  : `${dayWord(peakEndT, clock.issued)} ${fmtHour(peakEndT)}`}
              </button>
            </h3>
            <p className="worst__what">
              {peakCat.name}, up to <span className="num">{fmtPm(Math.max(...series.v.slice(worst[0], worst[1] + 1)))}</span> µg/m³ in the next 72 hours
            </p>
          </div>
        </div>
      </section>

      <section>
        <h3 className="section-title">Hour by hour</h3>
        <OutlookChart series={series} clock={clock} lead={lead} worst={worst} onLead={onLead} />
      </section>

      <section>
        <h3 className="section-title">Daily AQI</h3>
        <ul className="days">
          {days.map((d) => {
            const c = categoryOf(d.mean)
            return (
              <li key={d.label} className="day" style={{ '--cat': c.color, '--cat-ink': c.ink } as React.CSSProperties}>
                <span className="day__label">{d.label}</span>
                <span className="day__aqi num">{aqiValue(d.mean)}</span>
                <span className="day__name">{c.short}</span>
              </li>
            )
          })}
        </ul>
        <p className="footnote">DENR AQI from the forecast 24-hour mean{days[0]?.hours < 24 ? '; today uses the hours left' : ''}.</p>
      </section>

      <section className="details">
        <button type="button" className="details__toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
          <span>
            <span className="details__title">Forecast details</span>
            <span className="details__sub">Range, map confidence, model, and what drove this forecast</span>
          </span>
          <ChevronDown size={18} aria-hidden className="details__chev" />
        </button>
        {open && (
          <div className="details__body">
            <dl className="facts">
              <div>
                <dt>Likely range</dt>
                <dd>
                  <span className="num">{fmtPm(series.lo[lead])}–{fmtPm(series.hi[lead])}</span> µg/m³
                  <span className="muted"> · interval level to be set</span>
                </dd>
              </div>
              <div>
                <dt>Interval coverage (PICP)</dt>
                <dd className="muted">Pending evaluation</dd>
              </div>
              <div>
                <dt>Map confidence</dt>
                <dd>
                  {confWord} <span className="muted">· nearest station <span className="num">{(dist / 1000).toFixed(1)}</span> km. Relative only, not an error bound.</span>
                </dd>
              </div>
              <div>
                <dt>Grid cell</dt>
                <dd>
                  <span className="num">
                    {cell.col}, {cell.row}
                  </span>{' '}
                  <span className="muted">· 100 m · {fmtCoord(lng, lat)}</span>
                </dd>
              </div>
            </dl>

            <fieldset className="models">
              <legend className="field__label">Forecast model</legend>
              <div className="seg" role="radiogroup">
                {MODELS.map((m) => (
                  <label key={m.id} className="seg__opt" title={m.full}>
                    <input type="radio" name="model" value={m.id} checked={model === m.id} onChange={() => onModel(m.id)} />
                    <span>{m.name}</span>
                  </label>
                ))}
              </div>
              <p className="footnote">No model has been chosen as the best yet; evaluation results are pending.</p>
            </fieldset>

            <Explain key={`${cell.index}-${lead}-${model}`} field={field} cell={cell} lead={lead} model={model} />
          </div>
        )}
      </section>
    </div>
  )
}
