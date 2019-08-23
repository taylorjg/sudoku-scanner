import * as tf from '@tensorflow/tfjs'
import * as R from 'ramda'
import * as C from './constants'
import * as I from './image'
import * as U from './utils'
import * as CALC from './calculations'
import puzzles from '../data/puzzles.json'

const normaliseForGridCropping = ([x, y, w, h]) => {
  const normaliseX = value => value / (C.GRID_IMAGE_WIDTH - 1)
  const normaliseY = value => value / (C.GRID_IMAGE_HEIGHT - 1)
  return [
    normaliseY(y),
    normaliseX(x),
    normaliseY(y + h),
    normaliseX(x + w)
  ]
}

// tf.tidy ?
// options:
// - removeBlanks
// - createLabels
export const cropGridSquaresFromKnownGrid = (gridImageTensor, puzzleId, boundingBox, options = {}) => {
  const puzzle = puzzles.find(R.propEq('id', puzzleId))
  const gridSquares = CALC.calculateGridSquares(boundingBox)
  const blanksFilter = options.removeBlanks ? ({ isBlank }) => !isBlank : R.T
  const gridSquaresWithDetails = U.flattenInitialValues(puzzle.initialValues)
    .map((char, index) => ({
      isBlank: char === C.SPACE,
      digit: Number(char),
      gridSquare: gridSquares[index],
      index
    }))
    .filter(blanksFilter)
  const image = tf.stack(R.of(gridImageTensor.div(255)))
  const boxes = gridSquaresWithDetails.map(({ gridSquare }) => normaliseForGridCropping(gridSquare))
  const boxInd = R.repeat(0, boxes.length)
  const cropSize = [C.DIGIT_IMAGE_HEIGHT, C.DIGIT_IMAGE_WIDTH]
  const xs = tf.image.cropAndResize(image, boxes, boxInd, cropSize)
  const maybeLabels = options.createLabels
    ? { ys: options.createLabels(gridSquaresWithDetails) }
    : undefined
  return { xs, ...maybeLabels, gridImageTensor, gridSquaresWithDetails }
}

// tf.tidy ?
export const cropGridSquaresFromUnknownGrid = (gridImageTensor, boundingBox) => {
  const gridSquares = CALC.calculateGridSquares(boundingBox)
  const image = tf.stack(R.of(gridImageTensor.div(255)))
  const boxes = gridSquares.map(normaliseForGridCropping)
  const boxInd = R.repeat(0, boxes.length)
  const cropSize = [C.DIGIT_IMAGE_HEIGHT, C.DIGIT_IMAGE_WIDTH]
  return tf.image.cropAndResize(image, boxes, boxInd, cropSize)
}

export const createOneHotDigitLabels = gridSquaresWithDetails => {
  const oneBasedDigits = R.pluck('digit', gridSquaresWithDetails)
  const zeroBasedDigits = R.map(R.dec, oneBasedDigits)
  return tf.oneHot(zeroBasedDigits, 9)
}

export const createBlankOrNotBlankLabels = gridSquaresWithDetails =>
  tf.tensor1d(gridSquaresWithDetails.map(({ isBlank }) => isBlank ? 1 : 0))

const cropDigitsFromGrid = (item, gridImageTensor) =>
  cropGridSquaresFromKnownGrid(
    gridImageTensor,
    item.puzzleId,
    item.boundingBox,
    {
      removeBlanks: true,
      createLabels: createOneHotDigitLabels
    }
  )

const cropGridSquaresFromGrid = (item, gridImageTensor) =>
  cropGridSquaresFromKnownGrid(
    gridImageTensor,
    item.puzzleId,
    item.boundingBox,
    {
      removeBlanks: false,
      createLabels: createBlankOrNotBlankLabels
    }
  )

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
