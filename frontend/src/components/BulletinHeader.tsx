import { ChartLine, History, TriangleAlert } from 'lucide-react'
import logo from '../assets/brand/alapa-logo.png'
import type { Clock } from '../data/forecast'
import { fmtAgo, fmtDate, fmtTime, fmtYear } from '../lib/format'

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
const TITLE = 'Metro Manila PM2.5 forecast'

export function BulletinHeader({ clock, now, demo, past, onTests }: Props) {
  if (!clock) {
    return (
      <header className="masthead">
        <img className="masthead__logo" src={logo} alt="ALAPA" width={408} height={209} />
        <div className="masthead__issue">
          <h1 className="masthead__title">{TITLE}</h1>
        </div>
      </header>
    )
  }
  const stale = !past && now - clock.dataAsOf > STALE_MIN * 60_000
  const issued = fmtTime(clock.issued)
  return (
    <header className="masthead">
      <img className="masthead__logo" src={logo} alt="ALAPA" width={408} height={209} />
      <div className="masthead__issue">
        <h1 className="masthead__title">{TITLE}</h1>
        <p className="masthead__meta">
          Issued {fmtDate(clock.issued)}{past && `, ${fmtYear(clock.issued)}`}, <span className="num">{issued}</span>
          {!past && (
            <span className="masthead__next">
              {' '}· next <span className="num">{fmtTime(clock.issued + 3_600_000)}</span>
            </span>
          )}
          <span className="masthead__disclaimer"> · Not an official DENR advisory</span>
        </p>
      </div>
      <div className="masthead__side">
        {!past && (
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
        {demo && <span className="masthead__demo">Demo data</span>}
        {past && (
          <span className="masthead__pastwrap">
            <span className="masthead__past" tabIndex={0} aria-describedby="past-note">
              <History size={14} aria-hidden />
              Past forecast · recorded weather
            </span>
            <span className="masthead__tip" role="tooltip" id="past-note">
              A past forecast from the test period. It used the weather that was actually recorded, not a
              weather forecast. &ldquo;Now&rdquo; is its issue hour, <span className="num">{issued}</span>;
              readings after it are shown only to check it.
            </span>
          </span>
        )}
        {onTests && (
          <button type="button" className="masthead__btn" onClick={onTests} aria-haspopup="dialog">
            <ChartLine size={14} aria-hidden /> Forecast tests (O2)
          </button>
        )}
      </div>
    </header>
  )
}
