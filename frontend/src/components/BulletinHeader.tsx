import { ChartLine, History, TriangleAlert } from 'lucide-react'
import logo from '../assets/brand/alapa-logo.png'
import type { Clock } from '../data/forecast'
import { fmtAgo, fmtDate, fmtTime } from '../lib/format'

interface Props {
  /** Null until the forecast has loaded. */
  clock: Clock | null
  now: number
  demo: boolean
  /** A past forecast shown with what was measured after it, not the current bulletin. */
  past: boolean
  /** Opens O2's forecast tests; the button is left out when they are not published. */
  onTests?: () => void
}

const STALE_MIN = 90

export function BulletinHeader({ clock, now, demo, past, onTests }: Props) {
  if (!clock) {
    return (
      <header className="masthead">
        <img className="masthead__logo" src={logo} alt="ALAPA" width={408} height={209} />
        <div className="masthead__issue">
          <h1 className="masthead__title">Air Bulletin</h1>
        </div>
      </header>
    )
  }
  const stale = !past && now - clock.dataAsOf > STALE_MIN * 60_000
  return (
    <header className="masthead">
      <img className="masthead__logo" src={logo} alt="ALAPA" width={408} height={209} />
      <div className="masthead__issue">
        <h1 className="masthead__title">
          Air Bulletin <span className="masthead__no num">No. {clock.bulletinNo}</span>
        </h1>
        <p className="masthead__meta">
          <span className="masthead__scope">Metro Manila PM2.5 · </span>issued {fmtDate(clock.issued)}, <span className="num">{fmtTime(clock.issued)}</span>
          <span className="masthead__next">
            {' '}· next bulletin <span className="num">{fmtTime(clock.issued + 3_600_000)}</span>
          </span>
        </p>
      </div>
      {past ? (
        <p className="masthead__asof masthead__asof--past" role="status">
          <History size={16} aria-hidden />
          <span>
            Past forecast, so &ldquo;now&rdquo; is its issue hour.
            <span className="masthead__ago"> Readings after it are shown only to check it.</span>
          </span>
        </p>
      ) : (
        <p className={`masthead__asof${stale ? ' masthead__asof--stale' : ''}`} role={stale ? 'status' : undefined}>
          {stale && <TriangleAlert size={16} aria-hidden />}
          <span>
            Station data as of <span className="num">{fmtTime(clock.dataAsOf)}</span>
            <span className="masthead__ago">
              {' '}
              ({fmtAgo(clock.dataAsOf, now)}){stale ? '. Newer readings are late; the forecast may be out of date.' : ''}
            </span>
          </span>
        </p>
      )}
      <div className="masthead__notice">
        <p className="masthead__disclaimer">
          <span>Forecast by ALAPA, not an official DENR advisory.</span>
          {demo && <span className="masthead__demo">Demo data</span>}
          {past && <span className="masthead__demo">Past forecast</span>}
        </p>
        {onTests && (
          <button type="button" className="masthead__btn" onClick={onTests} aria-haspopup="dialog">
            <ChartLine size={14} aria-hidden /> Forecast tests (O2)
          </button>
        )}
      </div>
    </header>
  )
}
