import { LoaderCircle } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { BulletinHeader } from './components/BulletinHeader'
import { BulletinPanel, type LocStatus } from './components/BulletinPanel'
import { ForecastCone } from './components/ForecastCone'
import { MapKey } from './components/MapKey'
import { MapView } from './components/MapView'
import { ForecastField, makeClock, type ModelId, type Scenario } from './data/forecast'
import { cellAt, metersBetween, type CellRef, type LngLat } from './data/grid'
import { CITIES } from './data/places'

// Review hooks: ?scenario=clean|severe, ?state=loading|stale|denied, ?at=lng,lat, ?lead=n
const params = new URLSearchParams(window.location.search)
const scenario = (params.get('scenario') as Scenario) || 'normal'
const forced = params.get('state')
const presetAt = params.get('at')?.split(',').map(Number) as LngLat | undefined
const presetLead = Number(params.get('lead') ?? 0)

function placeNameFor(at: LngLat) {
  let best = CITIES[0]
  let d = Infinity
  for (const c of CITIES) {
    const dd = metersBetween(at, c.at)
    if (dd < d) {
      d = dd
      best = c
    }
  }
  return d < 1500 ? best.name : `Near ${best.name}`
}

export default function App() {
  const [now, setNow] = useState(() => Date.now())
  const hour = Math.floor(now / 3_600_000)
  const clock = useMemo(() => makeClock(hour * 3_600_000, forced === 'stale' ? 150 : 0), [hour])
  const [field, setField] = useState<ForecastField | null>(null)
  const [lead, setLead] = useState(Math.max(0, Math.min(71, presetLead)))
  const [model, setModel] = useState<ModelId>('gnn')
  const [showStations, setShowStations] = useState(false)
  const [showConfidence, setShowConfidence] = useState(false)
  const [place, setPlace] = useState<{ at: LngLat; cell: CellRef } | null>(null)
  const [locStatus, setLocStatus] = useState<LocStatus>('asking')
  const [flyTo, setFlyTo] = useState<{ at: LngLat; zoom: number; key: number } | null>(null)

  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 60_000)
    return () => window.clearInterval(t)
  }, [])

  // Building the grid takes a moment; let the shell paint first.
  useEffect(() => {
    if (forced === 'loading') return
    const t = window.setTimeout(() => setField(new ForecastField(clock, scenario)), 30)
    return () => window.clearTimeout(t)
  }, [clock])

  const choose = useCallback((at: LngLat, status: LocStatus, zoom?: number) => {
    const cell = cellAt(at[0], at[1])
    if (!cell) {
      setLocStatus(status === 'granted' ? 'outside' : status)
      setPlace(null)
      return false
    }
    setLocStatus(status)
    setPlace({ at, cell })
    if (zoom) setFlyTo({ at, zoom, key: Date.now() })
    return true
  }, [])

  const locate = useCallback(() => {
    if (forced === 'denied') {
      setLocStatus('denied')
      return
    }
    if (!('geolocation' in navigator)) {
      setLocStatus('unsupported')
      return
    }
    setLocStatus('asking')
    navigator.geolocation.getCurrentPosition(
      (pos) => choose([pos.coords.longitude, pos.coords.latitude], 'granted', 13),
      () => setLocStatus('denied'),
      { enableHighAccuracy: false, timeout: 10_000, maximumAge: 300_000 },
    )
  }, [choose])

  useEffect(() => {
    if (presetAt && presetAt.length === 2) choose(presetAt, 'picked', 12.5)
    else locate()
  }, [choose, locate])

  const onPick = useCallback((at: LngLat) => choose(at, 'picked'), [choose])
  const onCity = useCallback(
    (id: string) => {
      const c = CITIES.find((x) => x.id === id)
      if (c) choose(c.at, 'picked', 13)
    },
    [choose],
  )

  const share = useMemo(() => field?.metroShare(model) ?? null, [field, model])
  const spread = useMemo(() => field?.intervalSpread(model) ?? null, [field, model])

  return (
    <div className="app">
      <BulletinHeader clock={clock} now={now} />
      <main className="stage">
        <div className="stage__map">
          {field ? (
            <>
              <MapView
                field={field}
                lead={lead}
                model={model}
                showStations={showStations}
                showConfidence={showConfidence}
                selected={place?.cell ?? null}
                flyTo={flyTo}
                onPick={onPick}
              />
              <MapKey
                showStations={showStations}
                showConfidence={showConfidence}
                onStations={setShowStations}
                onConfidence={setShowConfidence}
              />
            </>
          ) : (
            <div className="loading" role="status">
              <LoaderCircle size={20} className="spin" aria-hidden />
              <span>Preparing the 100 m forecast for Metro Manila…</span>
            </div>
          )}
        </div>
        {field ? (
          <BulletinPanel
            field={field}
            clock={clock}
            cell={place?.cell ?? null}
            placeName={place ? placeNameFor(place.at) : ''}
            locStatus={locStatus}
            lead={lead}
            model={model}
            onLead={setLead}
            onModel={setModel}
            onLocate={locate}
            onCity={onCity}
          />
        ) : (
          <aside className="panel panel--loading" aria-hidden>
            <div className="skeleton skeleton--title" />
            <div className="skeleton skeleton--block" />
            <div className="skeleton skeleton--line" />
            <div className="skeleton skeleton--line" />
          </aside>
        )}
        <div className="stage__cone">
          {field && share && spread ? (
            <ForecastCone share={share} spread={spread} clock={clock} lead={lead} onLead={setLead} />
          ) : (
            <div className="skeleton skeleton--cone" />
          )}
        </div>
      </main>
    </div>
  )
}
