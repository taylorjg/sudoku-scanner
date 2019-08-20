import * as R from 'ramda'

export const inset = (x, y, w, h, dx, dy) =>
  [x + dx, y + dy, w - 2 * dx, h - 2 * dy]

export function* calculateGridSquares(boundingBox) {
  const [bbx, bby, bbw, bbh] = boundingBox
  const w = bbw / 9
  const h = bbh / 9
  const dx = 2 // w / 10
  const dy = 2 // h / 10
  for (const row of R.range(0, 9)) {
    const y = bby + row * h
    for (const col of R.range(0, 9)) {
      const x = bbx + col * w
      yield inset(x, y, w, h, dx, dy)
    }
  }
}

export const calculateBoxes = boundingBox => {
  const [bbx, bby, bbw, bbh] = boundingBox
  const w = bbw / 3
  const h = bbh / 3
  const rows = R.range(0, 3)
  const cols = R.range(0, 3)
  return R.chain(row =>
    R.map(col => {
      const x = bbx + col * w
      const y = bby + row * h
      return [x, y, w, h]
    }, cols),
    rows)
}

export const calculateCorners = (boundingBox, delta = 20) => {
  const [bbx, bby, bbw, bbh] = boundingBox
  const left = ([x, y]) => [x - delta, y]
  const right = ([x, y]) => [x + delta, y]
  const up = ([x, y]) => [x, y - delta]
  const down = ([x, y]) => [x, y + delta]
  const tl = [bbx, bby]
  const tr = [bbx + bbw, bby]
  const br = [bbx + bbw, bby + bbh]
  const bl = [bbx, bby + bbh]
  return R.flatten([
    down(tl), tl, right(tl),
    left(tr), tr, down(tr),
    up(br), br, left(br),
    right(bl), bl, up(bl)
  ])
}

export const calculateBoxCorners = boundingBox => {
  const boxes = calculateBoxes(boundingBox)
  return R.chain(box => calculateCorners(box, 10), boxes)
}
