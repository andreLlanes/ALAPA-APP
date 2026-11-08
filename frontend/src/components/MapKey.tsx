import { Layers, RadioTower } from 'lucide-react'
import { useState } from 'react'
import { AQI } from '../data/aqi'

interface Props {
  showStations: boolean
  showConfidence: boolean
  onStations: (v: boolean) => void
  onConfidence: (v: boolean) => void
}

export function MapKey({ showStations, showConfidence, onStations, onConfidence }: Props) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mapkey" data-open={open}>
      <button type="button" className="mapkey__toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="mapkey__chips" aria-hidden>
          {AQI.map((c) => (
            <span key={c.key} className="swatch" style={{ background: c.color }} />
          ))}
        </span>
        {open ? 'Hide key' : 'Map key'}
      </button>
      <section className="legend" aria-label="DENR AQI levels">
        <h2 className="legend__title">
          DENR AQI level <span className="muted">PM2.5, µg/m³</span>
        </h2>
        <ol className="legend__list">
          {AQI.map((c) => (
            <li key={c.key} className="legend__item">
              <span className="legend__no" style={{ background: c.color, color: c.ink }} aria-hidden>
                {c.level}
              </span>
              <span className="legend__name">{c.name}</span>
              <span className="legend__range num">{c.key === 'emergency' ? `${c.lo}+` : `${c.lo === 0 ? 0 : c.lo}–${c.hi}`}</span>
            </li>
          ))}
        </ol>
        <p className="legend__note">Each square is a 100 m cell. City badges show the level around each city center. The dashed line marks ALAPA's coverage.</p>
      </section>
      <section className="layers" aria-label="Map layers">
        <label className="toggle">
          <input type="checkbox" checked={showConfidence} onChange={(e) => onConfidence(e.target.checked)} />
          <Layers size={15} aria-hidden />
          <span>
            Map confidence
            <span className="toggle__sub">Hatched where stations are far</span>
          </span>
        </label>
        <label className="toggle">
          <input type="checkbox" checked={showStations} onChange={(e) => onStations(e.target.checked)} />
          <RadioTower size={15} aria-hidden />
          <span>
            Monitoring stations
            <span className="toggle__sub">Filled dot is reference-grade</span>
          </span>
        </label>
      </section>
    </div>
  )
}
