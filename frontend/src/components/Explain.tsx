import { LoaderCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { ForecastField, ModelId } from '../data/forecast'
import type { CellRef } from '../data/grid'
import { fmtPm } from '../lib/format'

interface Props {
  field: ForecastField
  cell: CellRef
  lead: number
  model: ModelId
}

type State = { kind: 'idle' } | { kind: 'loading' } | { kind: 'done'; data: ReturnType<ForecastField['explain']> }

/** SHAP attributions are computed on request, never precomputed for the grid. */
export function Explain({ field, cell, lead, model }: Props) {
  const [state, setState] = useState<State>({ kind: 'idle' })

  useEffect(() => {
    if (state.kind !== 'loading') return
    const t = window.setTimeout(() => setState({ kind: 'done', data: field.explain(cell.index, lead, model) }), 900)
    return () => window.clearTimeout(t)
  }, [state.kind, field, cell.index, lead, model])

  if (state.kind === 'idle') {
    return (
      <div className="explain">
        <h4 className="field__label">What the model relied on</h4>
        <p className="footnote">Computed for this place and hour when you ask. Takes a few seconds.</p>
        <button type="button" className="btn" onClick={() => setState({ kind: 'loading' })}>
          Explain this forecast
        </button>
      </div>
    )
  }

  if (state.kind === 'loading') {
    return (
      <div className="explain" aria-busy="true">
        <h4 className="field__label">What the model relied on</h4>
        <p className="explain__loading">
          <LoaderCircle size={16} className="spin" aria-hidden /> Computing feature attributions…
        </p>
      </div>
    )
  }

  const { data } = state
  const max = Math.max(...data.items.map((i) => Math.abs(i.contribution)), 1)
  return (
    <div className="explain">
      <h4 className="field__label">What the model relied on</h4>
      <p className="footnote">
        Starting from a typical <span className="num">{fmtPm(data.baseline)}</span> µg/m³, these inputs moved the forecast to{' '}
        <span className="num">{fmtPm(data.value)}</span>.
      </p>
      <ul className="shap">
        {data.items.map((i) => {
          const up = i.contribution >= 0
          const w = (Math.abs(i.contribution) / max) * 50
          return (
            <li key={i.key} className="shap__row">
              <span className="shap__label">{i.label}</span>
              <span className="shap__bar" aria-hidden>
                <span
                  className={`shap__fill ${up ? 'shap__fill--up' : 'shap__fill--down'}`}
                  style={up ? { left: '50%', width: `${w}%` } : { right: '50%', width: `${w}%` }}
                />
              </span>
              <span className="shap__val num">
                {up ? '+' : '−'}
                {fmtPm(Math.abs(i.contribution))}
              </span>
            </li>
          )
        })}
      </ul>
      <p className="footnote">
        These show how the model used its inputs, not what causes pollution. Synthetic example until the model service is connected.
      </p>
    </div>
  )
}
