import { LoaderCircle } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { BulletinHeader } from './components/BulletinHeader'
import { BulletinPanel, type LocStatus } from './components/BulletinPanel'
import { ForecastCone } from './components/ForecastCone'
import { MapKey } from './components/MapKey'
import { MapView } from './components/MapView'
import { O2Tests } from './components/O2Tests'
import { DemoField, makeClock, type ForecastField, type ModelId, type Scenario } from './data/forecast'
import { loadForecast } from './data/gridField'
import { cellAt, metersBetween, type CellRef, type LngLat } from './data/grid'
import { loadO2Tests, type O2Tests as O2TestsData } from './data/o2Tests'
import { CITIES } from './data/places'

// Review hooks: ?demo (synthetic data even when a map is published), ?scenario=clean|severe
// (demo only), ?state=loading|stale|denied, ?at=lng,lat, ?lead=n, ?tests (opens O2's forecast tests)
const params = new URLSearchParams(window.location.search)
const demoOnly = params.has('demo')
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
  const [field, setField] = useState<ForecastField | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const clock = field?.clock ?? null
  const [lead, setLead] = useState(Math.max(0, Math.min(71, presetLead)))
  const [chosenModel, setModel] = useState<ModelId>('')
  // The chosen model while the field has it, otherwise the field's default.
  const model = field?.models.some((m) => m.id === chosenModel) ? chosenModel : (field?.defaultModel ?? '')
  // Station dots start on for a past forecast, where they show what was measured.
  const [stationsChoice, setShowStations] = useState<boolean | null>(null)
  const showStations = stationsChoice ?? !!field?.readings?.backtest
  const [showConfidence, setShowConfidence] = useState(false)
  const [place, setPlace] = useState<{ at: LngLat; cell: CellRef; station?: string } | null>(null)
  const [locStatus, setLocStatus] = useState<LocStatus>('asking')
  const [flyTo, setFlyTo] = useState<{ at: LngLat; zoom: number; key: number } | null>(null)
  // O2's forecast tests, a separate file from the map; the header leaves the button out without it.
  const [o2, setO2] = useState<O2TestsData | null>(null)
  const [testsOpen, setTestsOpen] = useState(params.has('tests'))

  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 60_000)
    return () => window.clearInterval(t)
  }, [])

  useEffect(() => {
    let live = true
    loadO2Tests().then((d) => live && setO2(d))
    return () => {
      live = false
    }
  }, [])

  // The published forecast map, checked again every hour; demo data when none is published.
  // Building the grid takes a moment, so let the shell paint first.
  useEffect(() => {
    if (forced === 'loading') return
    let live = true
    const t = window.setTimeout(async () => {
      try {
        const published = demoOnly ? null : await loadForecast()
        const next = published ?? new DemoField(makeClock(hour * 3_600_000, forced === 'stale' ? 150 : 0), scenario)
        if (live) {
          setField(next)
          setFailure(null)
        }
      } catch (err) {
        if (live) setFailure(err instanceof Error ? err.message : String(err))
      }
    }, 30)
    return () => {
      live = false
      window.clearTimeout(t)
    }
  }, [hour])

  const choose = useCallback((at: LngLat, status: LocStatus, zoom?: number, station?: string) => {
    const cell = cellAt(at[0], at[1])
    if (!cell) {
      setLocStatus(status === 'granted' ? 'outside' : status)
      setPlace(null)
      return false
    }
    setLocStatus(status)
    setPlace({ at, cell, station })
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
  // The file's coordinates, not the tiled ones the map hands back, so the station keeps its cell.
  const onStation = useCallback(
    (id: string, at: LngLat) => choose(field?.readings?.stations.find((s) => s.id === id)?.at ?? at, 'picked', undefined, id),
    [choose, field],
  )
  // The station at the place: the one picked, or any station in the picked cell.
  const station = useMemo(() => {
    const r = field?.readings
    if (!r || !place) return null
    return r.stations.find((s) => (place.station ? s.id === place.station : s.cell === place.cell.index)) ?? null
  }, [field, place])
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
      <BulletinHeader
        clock={clock}
        now={now}
        demo={field?.provenance.demo ?? false}
        past={!!field?.readings?.backtest}
        onTests={o2 ? () => setTestsOpen(true) : undefined}
      />
      {o2 && testsOpen && <O2Tests data={o2} model={model} lead={lead} onClose={() => setTestsOpen(false)} />}
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
                onStation={onStation}
              />
              <MapKey
                showStations={showStations}
                showConfidence={showConfidence}
                readings={!!field.readings}
                onStations={setShowStations}
                onConfidence={setShowConfidence}
              />
            </>
          ) : failure ? (
            <div className="loading" role="alert">
              <span>Could not load the forecast map: {failure}</span>
            </div>
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
            clock={field.clock}
            cell={place?.cell ?? null}
            station={station}
            placeName={place ? placeNameFor(place.at) : ''}
            locStatus={locStatus}
            lead={lead}
            model={model}
            onLead={setLead}
            onModel={setModel}
            onLocate={locate}
            onCity={onCity}
            onStation={onStation}
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
            <ForecastCone share={share} spread={spread} clock={field.clock} lead={lead} onLead={setLead} />
          ) : (
            <div className="skeleton skeleton--cone" />
          )}
        </div>
      </main>
    </div>
  )
}
