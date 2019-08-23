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

// TODO: tf.tidy / tf.dispose
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

// TODO: tf.tidy / tf.dispose
export const cropGridSquaresFromUnknownGrid = (gridImageTensor, boundingBox) => {
  const gridSquares = CALC.calculateGridSquares(boundingBox)
  const image = tf.stack(R.of(gridImageTensor.div(255)))
  const boxes = gridSquares.map(normaliseForGridCropping)
  const boxInd = R.repeat(0, boxes.length)
  const cropSize = [C.DIGIT_IMAGE_HEIGHT, C.DIGIT_IMAGE_WIDTH]
  return tf.image.cropAndResize(image, boxes, boxInd, cropSize)
}

const createOneHotDigitLabels = gridSquaresWithDetails => {
  const oneBasedDigits = R.pluck('digit', gridSquaresWithDetails)
  const zeroBasedDigits = R.map(R.dec, oneBasedDigits)
  return tf.oneHot(zeroBasedDigits, 9)
}

const createBlankOrNotBlankLabels = gridSquaresWithDetails =>
  tf.tensor1d(gridSquaresWithDetails.map(({ isBlank }) => isBlank ? 1 : 0))

export const cropGridSquaresFromGrid = (item, gridImageTensor) =>
  cropGridSquaresFromKnownGrid(
    gridImageTensor,
    item.puzzleId,
    item.boundingBox,
    {
      removeBlanks: false,
      createLabels: createBlankOrNotBlankLabels
    }
  )

export const cropDigitsFromGrid = (item, gridImageTensor) =>
  cropGridSquaresFromKnownGrid(
    gridImageTensor,
    item.puzzleId,
    item.boundingBox,
    {
      removeBlanks: true,
      createLabels: createOneHotDigitLabels
    }
  )

// TODO: tf.tidy / tf.dispose
const loadKnownGridsAndCropGridSquares = async (data, options) => {
  const urls = R.pluck('url', data)
  const promises = urls.map(I.loadImage)
  const gridImageTensorsArray = await Promise.all(promises)
  const perGridResults = gridImageTensorsArray.map((gridImageTensor, index) => {
    const item = data[index]
    return cropGridSquaresFromKnownGrid(
      gridImageTensor,
      item.puzzleId,
      item.boundingBox,
      options
    )
  })
  const xss = R.pluck('xs', perGridResults)
  const yss = R.pluck('ys', perGridResults)
  const xs = tf.concat(xss)
  const ys = tf.concat(yss)
  return { xs, ys }
}

export const loadGridSquaresFromKnownGrids = data =>
  loadKnownGridsAndCropGridSquares(
    data,
    {
      removeBlanks: false,
      createLabels: createBlankOrNotBlankLabels
    })

export const loadDigitsFromKnownGrids = data =>
  loadKnownGridsAndCropGridSquares(
    data,
    {
      removeBlanks: true,
      createLabels: createOneHotDigitLabels
    })
