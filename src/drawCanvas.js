import * as tf from '@tensorflow/tfjs'
import * as R from 'ramda'
import * as CALC from './calculations'

export const drawBigRedCross = (ctx, boundingBox) => {
  const [x, y, w, h] = boundingBox
  const tl = [x, y]
  const tr = [x + w, y]
  const bl = [x, y + h]
  const br = [x + h, y + h]
  ctx.beginPath()
  ctx.moveTo(...tl)
  ctx.lineTo(...br)
  ctx.moveTo(...tr)
  ctx.lineTo(...bl)
  ctx.lineWidth = 5
  ctx.strokeStyle = 'red'
  ctx.stroke()
}

export const drawCorners = (canvas, corners, colour) => {
  const ctx = canvas.getContext('2d')
  const groupsOfCornerPoints = R.splitEvery(6, corners)
  groupsOfCornerPoints.forEach(([x1, y1, x2, y2, x3, y3]) => {
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.lineTo(x3, y3)
    ctx.strokeStyle = colour
    ctx.lineWidth = 2
    ctx.stroke()
  })
}

export const drawGridSquare = (canvas, gridSquare, colour) => {
  const ctx = canvas.getContext('2d')
  ctx.strokeStyle = colour
  ctx.lineWidth = 1
  ctx.strokeRect(...gridSquare)
}

export const drawGridSquares = (canvas, boundingBox, colour) => {
  const ctx = canvas.getContext('2d')
  for (const gridSquare of CALC.calculateGridSquares(boundingBox)) {
    ctx.strokeStyle = colour
    ctx.lineWidth = 1
    ctx.strokeRect(...gridSquare)
  }
}

export const drawBoundingBox = (canvas, boundingBox, colour) => {
  const ctx = canvas.getContext('2d')
  ctx.strokeStyle = colour
  ctx.lineWidth = 1
  ctx.strokeRect(...boundingBox)
}

export const drawGridImageTensor = async (parentElement, imageTensor) => {
  const canvas = document.createElement('canvas')
  canvas.setAttribute('class', 'grid-image')
  await tf.browser.toPixels(imageTensor, canvas)
  parentElement.appendChild(canvas)
  return canvas
}
