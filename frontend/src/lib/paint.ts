import { AQI, AQI_RGB } from '../data/aqi'
import type { ForecastField } from '../data/forecast'
import { CELL_COUNT, GRID } from '../data/grid'

const FIELD_ALPHA = 168

/** Paints one hour of the 100 m field, one pixel per cell, hard category bands. */
export function paintField(values: Float32Array, canvas: HTMLCanvasElement) {
  canvas.width = GRID.cols
  canvas.height = GRID.rows
  const ctx = canvas.getContext('2d')!
  const img = ctx.createImageData(GRID.cols, GRID.rows)
  const px = img.data
  for (let i = 0; i < CELL_COUNT; i++) {
    const v = values[i]
    const o = i * 4
    if (Number.isNaN(v)) {
      px[o + 3] = 0
      continue
    }
    let k = 0
    while (k < AQI.length - 1 && v > AQI[k].hi) k++
    const c = AQI_RGB[k]
    px[o] = c[0]
    px[o + 1] = c[1]
    px[o + 2] = c[2]
    px[o + 3] = FIELD_ALPHA
  }
  ctx.putImageData(img, 0, 0)
  return canvas
}

/**
 * Hatching where the spatialized surface is least certain. Dense lines far from
 * stations, sparse lines in between, nothing near the network.
 */
export function paintConfidence(field: ForecastField, canvas: HTMLCanvasElement) {
  const S = 4
  canvas.width = GRID.cols * S
  canvas.height = GRID.rows * S
  const ctx = canvas.getContext('2d')!

  const layer = (min: number, max: number, gap: number, alpha: number) => {
    const mask = document.createElement('canvas')
    mask.width = GRID.cols
    mask.height = GRID.rows
    const mctx = mask.getContext('2d')!
    const img = mctx.createImageData(GRID.cols, GRID.rows)
    for (let i = 0; i < CELL_COUNT; i++) {
      if (field.isWater(i)) continue
      const c = field.confidence(i)
      if (c >= min && c < max) img.data[i * 4 + 3] = 255
    }
    mctx.putImageData(img, 0, 0)

    const tmp = document.createElement('canvas')
    tmp.width = canvas.width
    tmp.height = canvas.height
    const t = tmp.getContext('2d')!
    t.imageSmoothingEnabled = false
    t.drawImage(mask, 0, 0, tmp.width, tmp.height)
    t.globalCompositeOperation = 'source-in'
    t.strokeStyle = `rgba(18, 22, 31, ${alpha})`
    t.lineWidth = 1.2
    t.beginPath()
    for (let d = -tmp.height; d < tmp.width + tmp.height; d += gap) {
      t.moveTo(d, 0)
      t.lineTo(d + tmp.height, tmp.height)
    }
    t.stroke()
    ctx.drawImage(tmp, 0, 0)
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  layer(0, 0.25, 7, 0.5)
  layer(0.25, 0.55, 14, 0.36)
  return canvas
}
