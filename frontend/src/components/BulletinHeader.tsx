import { TriangleAlert } from 'lucide-react'
import logo from '../assets/brand/alapa-logo.png'
import type { Clock } from '../data/forecast'
import { fmtAgo, fmtDate, fmtTime } from '../lib/format'

interface Props {
  clock: Clock
  now: number
}

const STALE_MIN = 90

export function BulletinHeader({ clock, now }: Props) {
  const stale = now - clock.dataAsOf > STALE_MIN * 60_000
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
      <p className="masthead__notice">
        <span>Forecast by ALAPA, not an official DENR advisory.</span>
        <span className="masthead__demo">Demo data</span>
      </p>
    </header>
  )
}
