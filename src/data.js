import * as tf from '@tensorflow/tfjs'
import * as R from 'ramda'
import * as C from './constants'
import * as CALC from './calculations'
import * as DC from './drawCanvas'
import * as I from './image'
import * as U from './utils'
import puzzles from '../data/puzzles.json'

// tf.tidy ?
export const loadGridData = async (data, parentElement) => {
  U.deleteChildren(parentElement)
  const targets = data.map(item => item.boundingBox)
  const urls = R.pluck('url', data)
  const promises = urls.map(I.loadImage)
  const imageTensors = await Promise.all(promises)
  imageTensors.forEach(async (imageTensor, index) => {
    const canvas = await DC.drawGridImageTensor(parentElement, imageTensor)
    canvas.setAttribute('title', data[index].url)
    const target = targets[index]
    DC.drawBoundingBox(canvas, target, 'blue')
  })
  const xs = tf.stack(imageTensors)
  const ys = tf.tensor2d(targets)
  return { xs, ys }
}

const normaliseX = x => x / (C.GRID_IMAGE_WIDTH - 1)
const normaliseY = y => y / (C.GRID_IMAGE_HEIGHT - 1)

// tf.tidy ?
export const cropGridSquaresFromGridGivenBoundingBox = (gridImageTensor, puzzleId, boundingBox, options = {}) => {
  const puzzle = puzzles.find(R.propEq('id', puzzleId))
  const gridSquares = Array.from(CALC.calculateGridSquares(boundingBox))
  const flattenedInitialValues = Array.from(puzzle.initialValues.join(''))
  const blanksFilter = options.removeBlanks ? ({ isBlank }) => !isBlank : R.T
  const digitsFilter = options.removeDigits ? ({ isBlank }) => isBlank : R.T
  const gridSquaresWithDetails = flattenedInitialValues
    .map((ch, index) => ({
      isBlank: ch === ' ',
      digit: Number(ch),
      gridSquare: gridSquares[index],
      index
    }))
    .filter(blanksFilter)
    .filter(digitsFilter)
  const image = tf.stack([gridImageTensor.div(255)])
  const boxes = gridSquaresWithDetails.map(({ gridSquare: [x, y, w, h] }) =>
    [
      normaliseY(y),
      normaliseX(x),
      normaliseY(y + h),
      normaliseX(x + w)
    ]
  )
  const boxInd = Array(boxes.length).fill(0)
  const cropSize = [C.DIGIT_IMAGE_HEIGHT, C.DIGIT_IMAGE_WIDTH]
  const xs = tf.image.cropAndResize(image, boxes, boxInd, cropSize)
  return { xs, gridSquaresWithDetails }
}

// tf.tidy ?
export const cropGridSquaresFromUnknownGrid = (gridImageTensor, boundingBox) => {
  const gridSquares = Array.from(CALC.calculateGridSquares(boundingBox))
  const image = tf.stack([gridImageTensor.div(255)])
  const boxes = gridSquares.map(([x, y, w, h]) =>
    [
      normaliseY(y),
      normaliseX(x),
      normaliseY(y + h),
      normaliseX(x + w)
    ]
  )
  const boxInd = Array(boxes.length).fill(0)
  const cropSize = [C.DIGIT_IMAGE_HEIGHT, C.DIGIT_IMAGE_WIDTH]
  return tf.image.cropAndResize(image, boxes, boxInd, cropSize)
}

// tf.tidy ?
export const cropGridSquaresFromGridCommon = (item, gridImageTensor, options = {}) => {
  const { puzzleId, boundingBox } = item
  const puzzle = puzzles.find(p => p.id === puzzleId)
  const gridSquares = Array.from(CALC.calculateGridSquares(boundingBox))
  const flattenedInitialValues = Array.from(puzzle.initialValues.join(''))
  const blanksFilter = options.removeBlanks ? ({ isBlank }) => !isBlank : R.T
  const digitsFilter = options.removeDigits ? ({ isBlank }) => isBlank : R.T
  const gridSquaresWithDetails = flattenedInitialValues
    .map((ch, index) => ({
      isBlank: ch === ' ',
      digit: Number(ch),
      gridSquare: gridSquares[index],
      index
    }))
    .filter(blanksFilter)
    .filter(digitsFilter)
  const image = tf.stack([gridImageTensor.div(255)])
  const boxes = gridSquaresWithDetails.map(({ gridSquare: [x, y, w, h] }) =>
    [
      normaliseY(y),
      normaliseX(x),
      normaliseY(y + h),
      normaliseX(x + w)
    ]
  )
  const boxInd = Array(boxes.length).fill(0)
  const cropSize = [C.DIGIT_IMAGE_HEIGHT, C.DIGIT_IMAGE_WIDTH]
  const xs = tf.image.cropAndResize(image, boxes, boxInd, cropSize)
  return { xs, gridSquaresWithDetails }
}

// tf.tidy ?
// ys: one-hots
export const cropDigitsFromGrid = (item, gridImageTensor) => {
  const options = { removeBlanks: true }
  const { xs, gridSquaresWithDetails } = cropGridSquaresFromGridCommon(item, gridImageTensor, options)
  const oneBasedDigits = R.pluck('digit', gridSquaresWithDetails)
  const zeroBasedDigits = R.map(R.dec, oneBasedDigits)
  const ys = tf.oneHot(zeroBasedDigits, 9)
  return { xs, ys, item, gridImageTensor, gridSquaresWithDetails }
}

// tf.tidy ?
// ys: 1 (blank) or 0 (digit)
export const cropGridSquaresFromGrid = (item, gridImageTensor) => {
  const { xs, gridSquaresWithDetails } = cropGridSquaresFromGridCommon(item, gridImageTensor)
  const ys = tf.tensor1d(gridSquaresWithDetails.map(({ isBlank }) => isBlank ? 1 : 0))
  return { xs, ys, item, gridImageTensor, gridSquaresWithDetails }
}

// tf.tidy ?
export const loadCroppedDataGrouped = async (data, cropFunction) => {
  const urls = R.pluck('url', data)
  const promises = urls.map(I.loadImage)
  const gridImageTensorsArray = await Promise.all(promises)
  return gridImageTensorsArray.map((gridImageTensor, index) => {
    const item = data[index]
    return cropFunction(item, gridImageTensor)
  })
}

export const loadGridSquaresGrouped = async data => loadCroppedDataGrouped(data, cropGridSquaresFromGrid)
export const loadDigitsGrouped = async data => loadCroppedDataGrouped(data, cropDigitsFromGrid)

// tf.tidy ?
const flattenGroupedData = async (data, loader) => {
  const groupedData = await loader(data)
  const xss = R.pluck('xs', groupedData)
  const yss = R.pluck('ys', groupedData)
  const xs = tf.concat(xss)
  const ys = tf.concat(yss)
  return { xs, ys }
}

export const loadDigitsFlat = data => flattenGroupedData(data, loadDigitsGrouped)
export const loadGridSquaresFlat = data => flattenGroupedData(data, loadGridSquaresGrouped)
