import {
  Map as MLMap,
  Marker,
  NavigationControl,
  Popup,
  ScaleControl,
  setWorkerUrl,
  type GeoJSONSource,
  type ImageSource,
  type MapMouseEvent,
} from 'maplibre-gl'
import type { Feature, FeatureCollection } from 'geojson'
import 'maplibre-gl/dist/maplibre-gl.css'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { useEffect, useRef } from 'react'
import { categoryOf } from '../data/aqi'
import type { ForecastField, ModelId } from '../data/forecast'
import { CELL_COUNT, GRID_BOUNDS, GRID_CORNERS, cellAt, cellPolygon, type CellRef, type LngLat } from '../data/grid'
import { CITIES } from '../data/places'
import { readingAt } from '../data/readings'
import { fmtPm } from '../lib/format'
import { paintConfidence, paintField } from '../lib/paint'

interface Props {
  field: ForecastField
  lead: number
  model: ModelId
  showStations: boolean
  showConfidence: boolean
  selected: CellRef | null
  flyTo: { at: LngLat; zoom: number; key: number } | null
  onPick: (at: LngLat) => void
  onStation: (id: string, at: LngLat) => void
}

/**
 * Station dots for one hour. With readings, a dot is filled with the AQI color of what the
 * station measured that hour (white when it has no reading) and carries the map's value at its
 * cell for the hover readout; without, the filled dot marks a reference-grade monitor.
 */
function stationData(f: ForecastField, lead: number, model: ModelId): FeatureCollection {
  const r = f.readings
  const features: Feature[] = r
    ? r.stations.map((s) => {
        const measured = readingAt(r, s, lead)
        const forecast = f.value(s.cell, lead, model)
        const has = Number.isFinite(measured)
        return {
          type: 'Feature',
          // Feature properties travel through the tiler: -1, not NaN, marks a missing number.
          properties: {
            id: s.id, name: s.name, reference: s.reference, inMap: s.inMap, has,
            fill: has ? categoryOf(measured).color : '#ffffff',
            measured: has ? measured : -1,
            forecast: Number.isFinite(forecast) ? forecast : -1,
          },
          geometry: { type: 'Point', coordinates: s.at },
        }
      })
    : f.stations.map((s) => ({
        type: 'Feature',
        properties: { id: s.id, name: s.name, reference: s.reference, inMap: true, has: true,
                      fill: s.reference ? '#12161f' : '#ffffff', measured: -1, forecast: -1 },
        geometry: { type: 'Point', coordinates: s.at },
      }))
  return { type: 'FeatureCollection', features }
}

/** Hover card for a station dot, built as DOM so no file content is read as HTML. */
function stationCard(p: Record<string, unknown>, withReadings: boolean) {
  const yes = (v: unknown) => v === true || v === 'true'
  const el = document.createElement('div')
  const line = (text: string, cls: string) => {
    const d = document.createElement('div')
    d.className = cls
    d.textContent = text
    el.appendChild(d)
  }
  line(String(p.name), 'station-pop__name')
  if (withReadings) {
    const measured = Number(p.measured)
    const forecast = Number(p.forecast)
    line(measured >= 0 ? `Measured ${fmtPm(measured)} µg/m³` : 'No reading this hour', 'station-pop__val')
    if (forecast >= 0) line(`Map ${fmtPm(forecast)} µg/m³`, 'station-pop__sub')
    line(yes(p.inMap) ? 'Used to make the map' : 'Not used by the map: an independent check', 'station-pop__sub')
  } else if (yes(p.reference)) {
    line('Reference-grade monitor', 'station-pop__sub')
  }
  return el
}

// Bundle MapLibre's worker ourselves; the build does not copy it otherwise.
setWorkerUrl(workerUrl)

// OpenFreeMap needs no key. The field slots under the basemap's water, roads, and
// labels, so the true coastline masks it and streets stay readable on top.
const FIELD_OPACITY = 0.6

export function MapView({ field, lead, model, showStations, showConfidence, selected, flyTo, onPick, onStation }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MLMap | null>(null)
  const ready = useRef(false)
  const front = useRef<'a' | 'b'>('a')
  const values = useRef(new Float32Array(CELL_COUNT))
  const badges = useRef(new Map<string, { el: HTMLDivElement; marker: Marker; priority: number }>())
  const pickRef = useRef(onPick)
  const stationRef = useRef(onStation)
  const latest = useRef({ field, lead, model, showStations, showConfidence, selected })

  useEffect(() => {
    pickRef.current = onPick
    stationRef.current = onStation
  }, [onPick, onStation])

  useEffect(() => {
    latest.current = { field, lead, model, showStations, showConfidence, selected }
  })

  // ---------------------------------------------------------------- mount
  useEffect(() => {
    if (!host.current) return
    const map = new MLMap({
      container: host.current,
      style: 'https://tiles.openfreemap.org/styles/positron',
      bounds: GRID_BOUNDS,
      fitBoundsOptions: { padding: 24 },
      maxBounds: [
        [GRID_BOUNDS[0][0] - 0.25, GRID_BOUNDS[0][1] - 0.2],
        [GRID_BOUNDS[1][0] + 0.25, GRID_BOUNDS[1][1] + 0.2],
      ],
      minZoom: 9,
      maxZoom: 17,
      attributionControl: { compact: true },
      dragRotate: false,
      pitchWithRotate: false,
    })
    map.touchZoomRotate.disableRotation()
    map.addControl(new NavigationControl({ showCompass: false }), 'top-right')
    map.addControl(new ScaleControl({ unit: 'metric', maxWidth: 110 }), 'bottom-right')
    mapRef.current = map

    map.on('load', () => {
      const { field: f, lead: l, model: m } = latest.current
      const layers = map.getStyle().layers
      const underWater = layers.find((ly) => ly.id === 'water')?.id
      const underLabels = layers.find((ly) => ly.type === 'symbol')?.id
      map.setPaintProperty('background', 'background-color', '#eef1f5')
      // The badges name the 17 NCR cities; drop the basemap's own labels for those
      // names only, so towns outside Metro Manila stay labeled inside the grid.
      const badged = CITIES.flatMap((c) => [c.name, `${c.name} City`, `City of ${c.name}`])
      for (const id of ['label_city', 'label_city_capital', 'label_town', 'label_village', 'label_other']) {
        if (!map.getLayer(id)) continue
        const base = map.getFilter(id)
        const notBadged = ['!', ['in', ['coalesce', ['get', 'name:en'], ['get', 'name_en'], ['get', 'name'], ''], ['literal', badged]]]
        map.setFilter(id, (base ? ['all', base, notBadged] : notBadged) as never)
      }
      const canvas = paintField(f.fillHour(l, m, values.current), document.createElement('canvas'))
      for (const id of ['a', 'b'] as const) {
        map.addSource(`field-${id}`, { type: 'image', coordinates: GRID_CORNERS })
        ;(map.getSource(`field-${id}`) as ImageSource).updateImage({ image: canvas })
        map.addLayer({
          id: `field-${id}`,
          type: 'raster',
          source: `field-${id}`,
          paint: {
            'raster-opacity': id === 'a' ? FIELD_OPACITY : 0,
            'raster-opacity-transition': { duration: 260, delay: 0 },
            'raster-resampling': 'nearest',
            'raster-fade-duration': 0,
          },
        }, underWater)
      }

      map.addSource('confidence', { type: 'image', coordinates: GRID_CORNERS })
      ;(map.getSource('confidence') as ImageSource).updateImage({
        image: paintConfidence(f, m, document.createElement('canvas')),
      })
      map.addLayer({
        id: 'confidence',
        type: 'raster',
        source: 'confidence',
        layout: { visibility: latest.current.showConfidence ? 'visible' : 'none' },
        paint: { 'raster-opacity': 0.85, 'raster-fade-duration': 0 },
      }, underLabels)

      map.addSource('coverage', {
        type: 'geojson',
        data: {
          type: 'Feature',
          properties: {},
          geometry: { type: 'LineString', coordinates: [...GRID_CORNERS, GRID_CORNERS[0]] },
        },
      })
      map.addLayer({
        id: 'coverage',
        type: 'line',
        source: 'coverage',
        paint: { 'line-color': '#0b3171', 'line-width': 1.5, 'line-dasharray': [4, 3], 'line-opacity': 0.7 },
      }, underLabels)

      map.addSource('stations', { type: 'geojson', data: stationData(f, l, m) })
      const dotRadius = f.readings ? ([9, 3.5, 14, 7] as const) : ([9, 2.5, 14, 5.5] as const)
      const stationsVisible = latest.current.showStations ? 'visible' : 'none'
      // A ring around a dot marks a station the map never used.
      map.addLayer({
        id: 'stations-ring',
        type: 'circle',
        source: 'stations',
        filter: ['==', ['get', 'inMap'], false],
        layout: { visibility: stationsVisible },
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], dotRadius[0], dotRadius[1] + 3, dotRadius[2], dotRadius[3] + 3.5],
          'circle-color': 'rgba(0,0,0,0)',
          'circle-stroke-color': '#12161f',
          'circle-stroke-width': 1.25,
        },
      })
      map.addLayer({
        id: 'stations',
        type: 'circle',
        source: 'stations',
        layout: { visibility: stationsVisible },
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], ...dotRadius],
          'circle-color': ['get', 'fill'],
          'circle-stroke-color': ['case', ['==', ['get', 'has'], true], '#12161f', '#8a93a3'],
          'circle-stroke-width': ['interpolate', ['linear'], ['zoom'], 9, 1, 14, 1.75],
        },
      })

      const card = new Popup({ closeButton: false, closeOnClick: false, offset: 12, className: 'station-pop' })
      map.on('mousemove', 'stations', (e) => {
        const feat = e.features?.[0]
        if (!feat || feat.geometry.type !== 'Point') return
        map.getCanvas().style.cursor = 'pointer'
        card.setLngLat(feat.geometry.coordinates as LngLat)
          .setDOMContent(stationCard(feat.properties, !!latest.current.field.readings))
          .addTo(map)
      })
      map.on('mouseleave', 'stations', () => {
        map.getCanvas().style.cursor = ''
        card.remove()
      })

      map.addSource('selected', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      map.addLayer({
        id: 'selected-halo',
        type: 'line',
        source: 'selected',
        paint: { 'line-color': '#ffffff', 'line-width': 5 },
      })
      map.addLayer({
        id: 'selected',
        type: 'line',
        source: 'selected',
        paint: { 'line-color': '#12161f', 'line-width': 2 },
      })

      for (const c of CITIES) {
        const el = document.createElement('div')
        el.className = 'city-badge'
        el.addEventListener('click', (e) => {
          e.stopPropagation()
          pickRef.current(c.at)
        })
        const marker = new Marker({ element: el, anchor: 'center' }).setLngLat(c.at).addTo(map)
        badges.current.set(c.id, { el, marker, priority: c.priority })
      }

      ready.current = true
      updateBadges()
      updateSelection()
      layoutBadges()
    })

    // A tap on a station dot picks that station; anywhere else picks the cell.
    map.on('click', (e: MapMouseEvent) => {
      const box: [[number, number], [number, number]] = [[e.point.x - 6, e.point.y - 6], [e.point.x + 6, e.point.y + 6]]
      const hit = ready.current && latest.current.showStations ? map.queryRenderedFeatures(box, { layers: ['stations'] })[0] : undefined
      if (hit && hit.geometry.type === 'Point') stationRef.current(String(hit.properties.id), hit.geometry.coordinates as LngLat)
      else pickRef.current([e.lngLat.lng, e.lngLat.lat])
    })
    map.on('moveend', layoutBadges)
    map.on('zoom', layoutBadges)

    function layoutBadges() {
      // Map-key cards and controls count as occupied space.
      const placed: DOMRect[] = [...(host.current?.parentElement?.querySelectorAll('.mapkey > *, .maplibregl-ctrl-group') ?? [])]
        .map((el) => el.getBoundingClientRect())
        .filter((r) => r.width > 0)
      const items = [...badges.current.values()].sort((a, b) => b.priority - a.priority)
      for (const b of items) {
        b.el.style.visibility = 'visible'
        const r = b.el.getBoundingClientRect()
        const hit = placed.some((p) => r.left < p.right + 4 && r.right + 4 > p.left && r.top < p.bottom + 3 && r.bottom + 3 > p.top)
        if (hit) b.el.style.visibility = 'hidden'
        else placed.push(r)
      }
    }

    function updateBadges() {
      const { field: f, lead: l, model: m } = latest.current
      for (const c of CITIES) {
        const b = badges.current.get(c.id)
        if (!b) continue
        const v = f.areaMean(c.at, l, m)
        const cat = categoryOf(v)
        b.el.style.setProperty('--cat', cat.color)
        b.el.style.setProperty('--cat-ink', cat.ink)
        const pm = Math.round(v)
        b.el.innerHTML = `<span class="city-badge__aqi">${cat.level}</span><span class="city-badge__name">${c.name}</span>`
        b.el.setAttribute('aria-label', `${c.name}: level ${cat.level}, ${cat.name}, area average ${pm} µg/m³`)
        b.el.title = `${c.name} area average · ${pm} µg/m³ · ${cat.name}`
      }
    }

    function updateSelection() {
      const src = map.getSource('selected') as GeoJSONSource | undefined
      const sel = latest.current.selected
      // A badge sitting on the selected cell steps aside so the outline shows.
      for (const c of CITIES) {
        const b = badges.current.get(c.id)
        const cc = cellAt(c.at[0], c.at[1])
        const near = !!sel && !!cc && Math.abs(cc.col - sel.col) <= 3 && Math.abs(cc.row - sel.row) <= 3
        b?.el.classList.toggle('city-badge--aside', near)
      }
      src?.setData({
        type: 'FeatureCollection',
        features: sel
          ? [{ type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [cellPolygon(sel)] } }]
          : [],
      })
    }

    ;(map as unknown as { __alapa: object }).__alapa = { updateBadges, updateSelection, layoutBadges }

    return () => {
      ready.current = false
      badges.current.clear()
      map.remove()
      mapRef.current = null
    }
  }, [])

  const helpers = () =>
    (mapRef.current as unknown as { __alapa?: { updateBadges(): void; updateSelection(): void; layoutBadges(): void } })
      ?.__alapa

  // ---------------------------------------------------------------- hour / model
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready.current) return
    const next = front.current === 'a' ? 'b' : 'a'
    const canvas = paintField(field.fillHour(lead, model, values.current), document.createElement('canvas'))
    ;(map.getSource(`field-${next}`) as ImageSource).updateImage({ image: canvas })
    map.setPaintProperty(`field-${next}`, 'raster-opacity', FIELD_OPACITY)
    map.setPaintProperty(`field-${front.current}`, 'raster-opacity', 0)
    front.current = next
    ;(map.getSource('stations') as GeoJSONSource).setData(stationData(field, lead, model))
    helpers()?.updateBadges()
  }, [field, lead, model])

  // Each model's kriging variance has its own shape, so the hatching follows the model.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready.current) return
    ;(map.getSource('confidence') as ImageSource).updateImage({
      image: paintConfidence(field, model, document.createElement('canvas')),
    })
  }, [field, model])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready.current) return
    for (const id of ['stations', 'stations-ring']) map.setLayoutProperty(id, 'visibility', showStations ? 'visible' : 'none')
  }, [showStations])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready.current) return
    map.setLayoutProperty('confidence', 'visibility', showConfidence ? 'visible' : 'none')
  }, [showConfidence])

  useEffect(() => {
    helpers()?.updateSelection()
  }, [selected])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !flyTo) return
    const go = () => map.easeTo({ center: flyTo.at, zoom: flyTo.zoom, duration: 700 })
    if (map.loaded()) go()
    else map.once('load', go)
  }, [flyTo])

  return <div ref={host} className="map" role="region" aria-label="PM2.5 forecast map of Metro Manila" />
}
