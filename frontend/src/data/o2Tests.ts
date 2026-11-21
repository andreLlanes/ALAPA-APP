// O2's error progression: the Metro Manila test RMSE at every hour ahead (1-72) of the station
// forecasts trained on Kaggle, per model, for the Manila-only run and the Bangkok and Los Angeles
// transfer runs, with persistence and climatology for reference. Written by
// O2/evals/app_export.py into data/o2_tests.json, apart from the map's manifest. It is O2's own
// test (70/15/15 split), separate from the map's check, and the two are never compared.

const DATA_URL = (import.meta.env.VITE_FORECAST_URL as string | undefined) ?? `${import.meta.env.BASE_URL}data/`

export const O2_MODELS = ['lstm', 'gnn', 'gbt'] as const
export type O2Model = (typeof O2_MODELS)[number]

export interface O2Line {
  id: 'mm' | 'bk' | 'la'
  label: string
  /** O2's variant name, e.g. param_full; "none" for the Manila-only run. */
  variant: string
  /** The variant in words, e.g. "parameter transfer, all layers". */
  how: string
  seeds: number
  /** RMSE (µg/m³) at hours ahead 1-72, the mean over seeds, and its confidence interval. */
  rmse: number[]
  lo: (number | null)[]
  hi: (number | null)[]
}

export interface O2Tests {
  version: 1
  source: string
  metric: 'rmse'
  settings: { split: string; completeness: number; longest_gap: number }
  test: { first_origin_utc: string; last_origin_utc: string; stations_scored: number; stations_with_windows: number }
  seeds: number
  leads: number
  baselines: { persistence: number[]; climatology: number[] }
  models: Record<O2Model, { lines: O2Line[] }>
}

const LEADS = 72

function complete(a: unknown, nullable = false): boolean {
  return (
    Array.isArray(a) &&
    a.length === LEADS &&
    a.every((v) => (nullable && v === null) || (typeof v === 'number' && Number.isFinite(v)))
  )
}

/** The checks the panel relies on; anything else means the file is not one we can draw. */
function valid(d: O2Tests): boolean {
  return (
    d.version === 1 &&
    d.leads === LEADS &&
    complete(d.baselines?.persistence) &&
    complete(d.baselines?.climatology) &&
    O2_MODELS.every((m) => {
      const lines = d.models?.[m]?.lines
      return (
        Array.isArray(lines) &&
        lines.length > 0 &&
        lines.every((l) => complete(l.rmse) && complete(l.lo, true) && complete(l.hi, true))
      )
    })
  )
}

/** O2's error progression, or null when it is not published (the app then leaves the panel out). */
export async function loadO2Tests(base = DATA_URL): Promise<O2Tests | null> {
  try {
    const res = await fetch(`${base}o2_tests.json`, { cache: 'no-cache' })
    // A dev server answers a missing file with the app's HTML page, so check the type too.
    if (!res.ok || !res.headers.get('content-type')?.includes('json')) return null
    const data = (await res.json()) as O2Tests
    return valid(data) ? data : null
  } catch {
    return null
  }
}
