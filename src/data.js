import * as tf from '@tensorflow/tfjs'
import * as R from 'ramda'
import * as C from './constants'
import * as CALC from './calculations'
import * as DC from './drawCanvas'
import puzzles from '../data/puzzles.json'

const deleteChildren = element => {
  while (element.firstChild) {
    element.removeChild(element.firstChild)
  }
}

// key: url, value: tensor3d
const GRID_IMAGE_CACHE = new Map()

export const loadImage = async url => {
  const existingImageTensor = GRID_IMAGE_CACHE.get(url)
  if (existingImageTensor) return existingImageTensor
  const promise = new Promise(resolve => {
    console.log(`Loading ${url}`)
    const image = new Image()
    image.onload = () => resolve(tf.browser.fromPixels(image, C.GRID_IMAGE_CHANNELS))
    image.src = url
  })
  const imageTensor = await promise
  GRID_IMAGE_CACHE.set(url, imageTensor)
  return imageTensor
}

export const loadGridData = async (data, elementId) => {
  const cornersArray = data.map(item => CALC.calculateBoxCorners(item.boundingBox))
  const urls = R.pluck('url', data)
  const promises = urls.map(loadImage)
  const imageTensors = await Promise.all(promises)
  const parentElement = document.getElementById(elementId)
  deleteChildren(parentElement)
  imageTensors.forEach(async (imageTensor, index) => {
    const canvas = await DC.drawGridImageTensor(parentElement, imageTensor)
    const corners = cornersArray[index]
    DC.drawCorners(canvas, corners, 'blue')
    // const boxCorners = CALC.calculateBoxCorners(data[index].boundingBox)
    // DC.drawCorners(canvas, boxCorners, 'blue')
  })
  const xs = tf.stack(imageTensors)
  const ys = tf.tensor2d(cornersArray, undefined, 'int32')
  return { xs, ys }
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
  const normaliseX = x => x / (C.GRID_IMAGE_WIDTH - 1)
  const normaliseY = y => y / (C.GRID_IMAGE_HEIGHT - 1)
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
  return { xs, puzzle, gridSquaresWithDetails }
}

// tf.tidy ?
// ys: one-hots
export const cropDigitsFromGrid = (item, gridImageTensor) => {
  const options = { removeBlanks: true }
  const { xs, puzzle, gridSquaresWithDetails } = cropGridSquaresFromGridCommon(item, gridImageTensor, options)
  const oneBasedDigits = R.pluck('digit', gridSquaresWithDetails)
  const zeroBasedDigits = R.map(R.dec, oneBasedDigits)
  const ys = tf.oneHot(zeroBasedDigits, 9)
  return { xs, ys, item, puzzle, gridImageTensor, gridSquaresWithDetails }
}

// tf.tidy ?
// ys: 1 (blank) or 0 (digit)
export const cropGridSquaresFromGrid = (item, gridImageTensor) => {
  const { xs, puzzle, gridSquaresWithDetails } = cropGridSquaresFromGridCommon(item, gridImageTensor)
  const ys = tf.tensor1d(gridSquaresWithDetails.map(({ isBlank }) => isBlank ? 1 : 0))
  return { xs, ys, item, puzzle, gridImageTensor, gridSquaresWithDetails }
}

// tf.tidy ?
export const loadCroppedDataGrouped = async (data, cropFunction) => {
  const urls = R.pluck('url', data)
  const promises = urls.map(loadImage)
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
