import { useMemo } from 'react'
import type { ForecastField, ModelId } from '../data/forecast'
import { score, type Score, type StationReadings } from '../data/readings'
import { fmtPm, fmtSigned } from '../lib/format'

// A past forecast checked against what the stations measured during its 72 hours. The map is
// compared at the 100 m cell each station sits in; nothing is checked between stations, where
// nothing was measured.

const pct = (v: number) => (Number.isNaN(v) ? '–' : `${Math.round(v * 100)}%`)

/** Every station with readings, split by whether its forecast made the map. */
export function AreaCheck({ field, model }: { field: ForecastField; model: ModelId }) {
  const rows = useMemo(() => {
    const all = field.readings?.stations ?? []
    return [
      { key: 'used', label: 'Used by the map', s: score(field, model, all.filter((s) => s.inMap)) },
      { key: 'unused', label: 'Not used by the map', s: score(field, model, all.filter((s) => !s.inMap)) },
    ].filter((r) => r.s.hours > 0)
  }, [field, model])
  if (!rows.length) return null

  return (
    <section className="check" aria-label="How this past forecast did">
      <h3 className="section-title">How this past forecast did</h3>
      <p className="check__lede">
        The forecast at each station's 100 m cell against what the station measured, over the 72 hours.
      </p>
      <table className="check__table">
        <thead>
          <tr>
            <th scope="col" />
            <th scope="col">Stations</th>
            <th scope="col" title="Root-mean-square error, µg/m³">RMSE</th>
            <th scope="col" title="Mean of forecast minus reading, µg/m³">Bias</th>
            <th scope="col" title="Hours where forecast and reading fall in the same AQI level">Same level</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <th scope="row">{r.label}</th>
              <td className="num">{r.s.stations}</td>
              <td className="num">{fmtPm(r.s.rmse)}</td>
              <td className="num">{fmtSigned(r.s.bias)}</td>
              <td className="num">{pct(r.s.sameLevel)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="footnote">
        RMSE is the typical miss in µg/m³; bias is forecast minus reading on average. The map stays close to the
        forecasts it was made from, so the first row mostly tests the forecast and the second also tests the gaps
        between stations. Levels are hourly here; the DENR AQI uses 24-hour means.
      </p>
    </section>
  )
}

/** One station's 72 hours. */
export function StationCheck({ field, model, station }: { field: ForecastField; model: ModelId; station: StationReadings }) {
  const s: Score = useMemo(() => score(field, model, [station]), [field, model, station])
  return (
    <section className="check" aria-label="Forecast against this station's readings">
      <h3 className="section-title">Forecast vs measured here, 72 hours</h3>
      {s.hours === 0 ? (
        <p className="footnote">This station has no readings during the forecast hours.</p>
      ) : (
        <>
          <dl className="facts">
            <div>
              <dt>Typical miss</dt>
              <dd>
                <span className="num">{fmtPm(s.rmse)}</span> µg/m³ RMSE{' '}
                <span className="muted">
                  · mean absolute <span className="num">{fmtPm(s.mae)}</span>
                </span>
              </dd>
            </div>
            <div>
              <dt>Bias</dt>
              <dd>
                <span className="num">{fmtSigned(s.bias)}</span> µg/m³{' '}
                <span className="muted">· the forecast ran {s.bias >= 0 ? 'high' : 'low'} on average</span>
              </dd>
            </div>
            <div>
              <dt>Same AQI level</dt>
              <dd>
                <span className="num">{pct(s.sameLevel)}</span> of hours
              </dd>
            </div>
            <div>
              <dt>Hours checked</dt>
              <dd>
                <span className="num">{s.hours}</span> of 72 <span className="muted">· the rest had no reading</span>
              </dd>
            </div>
          </dl>
          <p className="footnote">
            {station.inMap
              ? "This station's own forecast went into the map, so this mostly tests the forecast."
              : 'The map never used this station, so this also tests how well the map fills the gaps between stations.'}
          </p>
        </>
      )}
    </section>
  )
}
