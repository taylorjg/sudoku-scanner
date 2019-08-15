import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import * as R from 'ramda'
import axios from 'axios'
import { createSvgElement, drawInitialGrid } from './svg'

import puzzles from '../data/puzzles.json'
import trainingData from '../data/training-data.json'
import trainingData2 from '../data/training-data-2.json'
import validationData from '../data/validation-data.json'
import testData from '../data/test-data.json'

const models = {
  grid: {
    model: undefined,
    trained: false
  },
  blanks: {
    model: undefined,
    trained: false
  },
  digits: {
    model: undefined,
    trained: false
  }
}

let imageData = undefined

const GRID_IMAGE_HEIGHT = 224
const GRID_IMAGE_WIDTH = 224
const GRID_IMAGE_CHANNELS = 1

const DIGIT_IMAGE_HEIGHT = 20
const DIGIT_IMAGE_WIDTH = 20
const DIGIT_IMAGE_CHANNELS = 1

const inset = (x, y, w, h, dx, dy) =>
  [x + dx, y + dy, w - 2 * dx, h - 2 * dy]

function* calculateGridSquares(boundingBox) {
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

const calculateCorners = boundingBox => {
  const [bbx, bby, bbw, bbh] = boundingBox
  const DELTA_X = 20
  const DELTA_Y = 20
  // const DELTA_X = bbw / 3
  // const DELTA_Y = bbh / 3
  const left = ([x, y]) => [x - DELTA_X, y]
  const right = ([x, y]) => [x + DELTA_X, y]
  const up = ([x, y]) => [x, y - DELTA_Y]
  const down = ([x, y]) => [x, y + DELTA_Y]
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

const drawBigRedCross = (ctx, boundingBox) => {
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

const drawCorners = (canvas, corners, colour) => {
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

// const drawGridSquares = (canvas, boundingBox, colour) => {
//   const ctx = canvas.getContext('2d')
//   for (const gridSquare of calculateGridSquares(boundingBox)) {
//     ctx.strokeStyle = colour
//     ctx.strokeRect(...gridSquare)
//   }
// }

// const drawBoundingBox = (canvas, boundingBox, colour) => {
//   const ctx = canvas.getContext('2d')
//   ctx.beginPath()
//   ctx.strokeStyle = colour
//   ctx.lineWidth = 1
//   ctx.strokeRect(...boundingBox)
// }

const drawGridImageTensor = async (parentElement, imageTensor) => {
  const canvas = document.createElement('canvas')
  canvas.setAttribute('class', 'grid-image')
  await tf.browser.toPixels(imageTensor, canvas)
  parentElement.appendChild(canvas)
  return canvas
}

const convertToGreyscale = imageData => {
  const w = imageData.width
  const h = imageData.height
  const numPixels = w * h
  const data = imageData.data
  const array = new Uint8ClampedArray(data.length)
  const bases = R.range(0, numPixels).map(index => index * 4)
  for (const base of bases) {
    const colourValues = data.slice(base, base + 4)
    const [r, g, b, a] = colourValues
    // https://imagemagick.org/script/command-line-options.php#colorspace
    // Gray = 0.212656*R+0.715158*G+0.072186*B
    const greyValue = 0.212656 * r + 0.715158 * g + 0.072186 * b
    array[base] = greyValue
    array[base + 1] = greyValue
    array[base + 2] = greyValue
    array[base + 3] = a
  }
  return new ImageData(array, w, h)
}

// key: url, value: tensor3d
const GRID_IMAGE_CACHE = new Map()

const loadImage = async url => {
  const existingImageTensor = GRID_IMAGE_CACHE.get(url)
  if (existingImageTensor) return existingImageTensor
  const promise = new Promise(resolve => {
    console.log(`Loading ${url}`)
    const image = new Image()
    image.onload = () => resolve(tf.browser.fromPixels(image, GRID_IMAGE_CHANNELS))
    image.src = url
  })
  const imageTensor = await promise
  GRID_IMAGE_CACHE.set(url, imageTensor)
  return imageTensor
}

const deleteChildren = element => {
  while (element.firstChild) {
    element.removeChild(element.firstChild)
  }
}

const loadGridData = async (data, elementId) => {
  const cornersArray = data.map(item => calculateCorners(item.boundingBox))
  const urls = R.pluck('url', data)
  const promises = urls.map(loadImage)
  const imageTensors = await Promise.all(promises)
  const parentElement = document.getElementById(elementId)
  deleteChildren(parentElement)
  imageTensors.forEach(async (imageTensor, index) => {
    const canvas = await drawGridImageTensor(parentElement, imageTensor)
    const corners = cornersArray[index]
    drawCorners(canvas, corners, 'blue')
  })
  const xs = tf.stack(imageTensors)
  const ys = tf.tensor2d(cornersArray, undefined, 'int32')
  return { xs, ys }
}

// tf.tidy ?
// Used by loadDigitsGrouped
// TODO: cropDigitsFromGrid and cropGridSquaresFromGrid are very similar.
//       I think we can combine these into a single function.
//       Use a flag to control whether blank grid squares are included in the results.
const cropDigitsFromGrid = (item, gridImageTensor) => {
  const { puzzleId, boundingBox } = item
  const gridSquares = Array.from(calculateGridSquares(boundingBox))
  const puzzle = puzzles.find(p => p.id === puzzleId)
  const flattenedInitialValues = Array.from(puzzle.initialValues.join(''))

  const digitsAndGridSquares = flattenedInitialValues
    .map((ch, index) => ({
      digit: Number(ch),
      index,
      gridSquare: gridSquares[index],
      isBlank: ch === ' '
    }))
    .filter(({ isBlank }) => !isBlank)

  const image = tf.stack([gridImageTensor.div(255)])
  const normaliseX = x => x / (GRID_IMAGE_WIDTH - 1)
  const normaliseY = y => y / (GRID_IMAGE_HEIGHT - 1)
  const boxes = digitsAndGridSquares.map(({ gridSquare: [x, y, w, h] }) =>
    [normaliseY(y), normaliseX(x), normaliseY(y + h), normaliseX(x + w)]
  )
  const boxInd = Array(boxes.length).fill(0)
  const cropSize = [DIGIT_IMAGE_HEIGHT, DIGIT_IMAGE_WIDTH]
  const xs = tf.image.cropAndResize(image, boxes, boxInd, cropSize)

  const oneBasedDigits = R.pluck('digit', digitsAndGridSquares)
  const zeroBasedDigits = R.map(R.dec, oneBasedDigits)
  const ys = tf.oneHot(zeroBasedDigits, 9)
  return { xs, ys, item, puzzle, gridImageTensor, digitsAndGridSquares }
}

// tf.tidy ?
// Used by loadGridSquaresGrouped
const cropGridSquaresFromGrid = (item, gridImageTensor) => {
  const { puzzleId, boundingBox } = item
  const gridSquares = Array.from(calculateGridSquares(boundingBox))
  const puzzle = puzzles.find(p => p.id === puzzleId)
  const flattenedInitialValues = Array.from(puzzle.initialValues.join(''))

  const gridSquaresWithDetails = flattenedInitialValues
    .map((ch, index) => ({
      digit: Number(ch),
      index,
      gridSquare: gridSquares[index],
      isBlank: ch === ' '
    }))

  const image = tf.stack([gridImageTensor.div(255)])
  const normaliseX = x => x / (GRID_IMAGE_WIDTH - 1)
  const normaliseY = y => y / (GRID_IMAGE_HEIGHT - 1)
  const boxes = gridSquaresWithDetails.map(({ gridSquare: [x, y, w, h] }) =>
    [normaliseY(y), normaliseX(x), normaliseY(y + h), normaliseX(x + w)]
  )
  const boxInd = Array(boxes.length).fill(0)
  const cropSize = [DIGIT_IMAGE_HEIGHT, DIGIT_IMAGE_WIDTH]
  const xs = tf.image.cropAndResize(image, boxes, boxInd, cropSize)

  const ys = tf.tensor1d(gridSquaresWithDetails.map(({ isBlank }) => isBlank ? 1 : 0))
  return { xs, ys, item, puzzle, gridImageTensor, gridSquaresWithDetails }
}

// tf.tidy ?
// Used by onPredictBlanks and onPredictBlanksDigits
const loadGridSquaresGrouped = async data => {
  const urls = R.pluck('url', data)
  const promises = urls.map(loadImage)
  const gridImageTensorsArray = await Promise.all(promises)
  return gridImageTensorsArray.map((gridImageTensor, index) => {
    const item = data[index]
    return cropGridSquaresFromGrid(item, gridImageTensor)
  })
}

// tf.tidy ?
// Used by onPredictDigits
const loadDigitsGrouped = async data => {
  const urls = R.pluck('url', data)
  const promises = urls.map(loadImage)
  const gridImageTensorsArray = await Promise.all(promises)
  return gridImageTensorsArray.map((gridImageTensor, index) => {
    const item = data[index]
    return cropDigitsFromGrid(item, gridImageTensor)
  })
}

// tf.tidy ?
const flattenGroupedData = async (data, loader) => {
  const groupedData = await loader(data)
  const xss = R.pluck('xs', groupedData)
  const yss = R.pluck('ys', groupedData)
  const xs = tf.concat(xss)
  const ys = tf.concat(yss)
  return { xs, ys }
}

const loadDigitsFlat = data => flattenGroupedData(data, loadDigitsGrouped)
const loadGridSquaresFlat = data => flattenGroupedData(data, loadGridSquaresGrouped)

const createGridModel = () => {

  const inputShape = [GRID_IMAGE_HEIGHT, GRID_IMAGE_WIDTH, GRID_IMAGE_CHANNELS]
  const conv2dArgs = {
    kernelSize: 7,
    filters: 8,
    // activation: 'sigmoid',
    activation: 'tanh',
    strides: 1,
    kernelInitializer: 'varianceScaling'
  }
  const maxPooling2dArgs = {
    poolSize: [2, 2],
    strides: [2, 2]
  }

  const model = tf.sequential()

  model.add(tf.layers.conv2d({ inputShape, ...conv2dArgs }))
  model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  model.add(tf.layers.conv2d({ ...conv2dArgs, activation: 'tanh' }))
  model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  model.add(tf.layers.conv2d({ ...conv2dArgs, activation: 'tanh' }))
  model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  model.add(tf.layers.conv2d({ ...conv2dArgs, activation: 'tanh' }))
  model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  model.add(tf.layers.conv2d({ ...conv2dArgs, activation: 'tanh' }))

  model.add(tf.layers.flatten())

  model.add(tf.layers.dense({ units: 256, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 24 }))

  model.summary()

  return model
}

const createBlanksModel = () => {
  const inputShape = [DIGIT_IMAGE_HEIGHT, DIGIT_IMAGE_WIDTH, DIGIT_IMAGE_CHANNELS]
  const conv2dArgs = {
    kernelSize: 5,
    filters: 32,
    activation: 'relu',
    strides: 1,
    kernelInitializer: 'varianceScaling'
  }
  const maxPooling2dArgs = {
    poolSize: [2, 2],
    strides: [2, 2]
  }

  const model = tf.sequential()

  model.add(tf.layers.conv2d({ inputShape, ...conv2dArgs }))
  model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  model.add(tf.layers.conv2d(conv2dArgs))

  model.add(tf.layers.flatten())

  model.add(tf.layers.dense({ units: 100, activation: 'sigmoid' }))
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }))

  model.summary()

  return model
}

const createDigitsModel = () => {

  const inputShape = [DIGIT_IMAGE_HEIGHT, DIGIT_IMAGE_WIDTH, DIGIT_IMAGE_CHANNELS]
  const conv2dArgs = {
    kernelSize: 5,
    filters: 32,
    activation: 'relu',
    strides: 1,
    kernelInitializer: 'varianceScaling'
  }
  const maxPooling2dArgs = {
    poolSize: [2, 2],
    strides: [2, 2]
  }

  const model = tf.sequential()

  model.add(tf.layers.conv2d({ inputShape, ...conv2dArgs }))
  model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  model.add(tf.layers.conv2d(conv2dArgs))

  model.add(tf.layers.flatten())

  model.add(tf.layers.dense({ units: 256, activation: 'relu' }))
  model.add(tf.layers.dropout({ rate: 0.2 }))
  model.add(tf.layers.dense({ units: 9, activation: 'softmax' }))

  model.summary()

  return model
}

const trainGrid = async model => {

  const combinedData = trainingData.concat(trainingData2).concat(validationData)
  tf.util.shuffle(combinedData)
  const { xs, ys } = await loadGridData(combinedData, 'gridTrainingData')

  model.compile({
    optimizer: 'rmsprop',
    // loss: 'meanSquaredError'
    loss: 'meanAbsoluteError'
  })

  const trainingSurface = tfvis.visor().surface({ tab: 'Grid', name: 'Model Training' })
  const customCallback = tfvis.show.fitCallbacks(
    trainingSurface,
    ['loss', 'val_loss'],
    {
      callbacks: ['onBatchEnd', 'onEpochEnd']
    })

  const params = {
    batchSize: 10,
    epochs: 50,
    shuffle: true,
    validationSplit: 0.20,
    callbacks: customCallback
  }

  return model.fit(xs, ys, params)
}

const trainBlanks = async model => {

  const combinedData = trainingData.concat(validationData)
  tf.util.shuffle(combinedData)
  const { xs, ys } = await loadGridSquaresFlat(combinedData)

  model.compile({
    optimizer: 'rmsprop',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  })

  const trainingSurface = tfvis.visor().surface({ tab: 'Blanks', name: 'Model Training' })
  const customCallback = tfvis.show.fitCallbacks(
    trainingSurface,
    ['loss', 'val_loss', 'acc', 'val_acc'],
    { callbacks: ['onBatchEnd', 'onEpochEnd'] }
  )

  const params = {
    batchSize: 100,
    epochs: 5,
    shuffle: true,
    validationSplit: 0.15,
    callbacks: customCallback
  }

  return model.fit(xs, ys, params)
}

const trainDigits = async model => {

  const combinedData = trainingData.concat(validationData)
  tf.util.shuffle(combinedData)
  const { xs, ys } = await loadDigitsFlat(combinedData)

  model.compile({
    optimizer: 'rmsprop',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  const trainingSurface = tfvis.visor().surface({ tab: 'Digits', name: 'Model Training' })
  const customCallback = tfvis.show.fitCallbacks(
    trainingSurface,
    ['loss', 'val_loss', 'acc', 'val_acc'],
    { callbacks: ['onBatchEnd', 'onEpochEnd'] }
  )

  const params = {
    batchSize: 10,
    epochs: 20,
    shuffle: true,
    validationSplit: 0.15,
    callbacks: customCallback
  }

  return model.fit(xs, ys, params)
}

const createVideoGuide = d =>
  createSvgElement('path', { d, class: 'video-guide' })

const drawGuides = () => {
  const svg = document.getElementById('video-guides')
  const wRect = svg.getBoundingClientRect().width
  const hRect = svg.getBoundingClientRect().height
  const wInset = wRect * 0.05
  const hInset = hRect * 0.05
  const wArm = wRect * 0.1
  const hArm = hRect * 0.1
  svg.appendChild(createVideoGuide(`M${wInset + wArm},${hInset} h${-wArm} v${hArm}`))
  svg.appendChild(createVideoGuide(`M${wRect - wInset - wArm},${hInset} h${wArm} v${hArm}`))
  svg.appendChild(createVideoGuide(`M${wInset + wArm},${hRect - hInset} h${-wArm} v${-hArm}`))
  svg.appendChild(createVideoGuide(`M${wRect - wInset - wArm},${hRect - hInset} h${wArm} v${-hArm}`))
}

const initialiseCamera = async () => {

  const videoElement = document.getElementById('video')
  const videoElementRect = videoElement.getBoundingClientRect()
  const capturedImageElement = document.getElementById('captured-image')
  capturedImageElement.width = videoElementRect.width
  capturedImageElement.height = videoElementRect.height
  const capturedImageElementContext = capturedImageElement.getContext('2d')
  const startBtn = document.getElementById('startBtn')
  const stopBtn = document.getElementById('stopBtn')
  const captureBtn = document.getElementById('captureBtn')
  const saveBtn = document.getElementById('saveBtn')
  const clearBtn = document.getElementById('clearBtn')
  const messageArea = document.getElementById('messageArea')

  const updateButtonState = () => {
    const playing = !!videoElement.srcObject
    startBtn.disabled = playing
    stopBtn.disabled = !playing
    captureBtn.disabled = !playing
    saveBtn.disabled = !imageData
    clearBtn.disabled = !imageData
    const allTrained = models.grid.trained && models.blanks.trained && models.digits.trained
    predictCaptureBtn.disabled = !(allTrained && imageData)
  }

  const onStart = async () => {
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: videoElementRect.width,
        height: videoElementRect.height,
        facingMode: 'environment'
      }
    })
    if (mediaStream) {
      videoElement.srcObject = mediaStream
      videoElement.play()
      updateButtonState()
    }
  }

  const onStop = () => {
    const mediaStream = videoElement.srcObject
    mediaStream.getVideoTracks()[0].stop()
    videoElement.srcObject = null
    updateButtonState()
  }

  const onCapture = async () => {
    const imageBitmap = await createImageBitmap(videoElement)
    capturedImageElementContext.drawImage(imageBitmap, 0, 0)
    const w = capturedImageElementContext.canvas.width
    const h = capturedImageElementContext.canvas.height
    imageData = capturedImageElementContext.getImageData(0, 0, w, h)
    onStop()
  }

  const onSave = async () => {
    const dataUrlRaw = capturedImageElement.toDataURL('image/png')
    const responseRaw = await axios.post('/api/saveRawImage', { dataUrl: dataUrlRaw })
    console.dir(responseRaw)
    messageArea.innerText = responseRaw.data

    const canvas = document.createElement('canvas')
    canvas.width = GRID_IMAGE_WIDTH
    canvas.height = GRID_IMAGE_HEIGHT
    const imageTensor = normaliseGridImage(imageData)
    await tf.browser.toPixels(imageTensor, canvas)
    const dataUrlNormalised = canvas.toDataURL('image/png')
    const responseNormalised = await axios.post('/api/saveNormalisedImage', { dataUrl: dataUrlNormalised })
    console.dir(responseNormalised)
  }

  const onClear = () => {
    capturedImageElementContext.clearRect(0, 0, capturedImageElement.width, capturedImageElement.height)
    imageData = undefined
    messageArea.innerText = ''
    updateButtonState()
  }

  startBtn.addEventListener('click', onStart)
  stopBtn.addEventListener('click', onStop)
  captureBtn.addEventListener('click', onCapture)
  saveBtn.addEventListener('click', onSave)
  clearBtn.addEventListener('click', onClear)

  updateButtonState()
}

const onTrainGrid = async () => {
  try {
    trainGridBtn.disabled = true
    models.grid.model = createGridModel()
    await trainGrid(models.grid.model)
    models.grid.trained = true // eslint-disable-line
    updateButtonStates()
  } finally {
    trainGridBtn.disabled = false
  }
}

const onTrainBlanks = async () => {
  try {
    trainBlanksBtn.disabled = true
    models.blanks.model = createBlanksModel()
    await trainBlanks(models.blanks.model)
    models.blanks.trained = true // eslint-disable-line
    updateButtonStates()
  } finally {
    trainBlanksBtn.disabled = false
  }
}

const onTrainDigits = async () => {
  try {
    trainDigitsBtn.disabled = true
    models.digits.model = createDigitsModel()
    await trainDigits(models.digits.model)
    models.digits.trained = true // eslint-disable-line
    updateButtonStates()
  } finally {
    trainDigitsBtn.disabled = false
  }
}

const normaliseGridImage = imageData => {
  const imageDataGreyscale = convertToGreyscale(imageData)
  const imageTensorGreyscale = tf.browser.fromPixels(imageDataGreyscale, GRID_IMAGE_CHANNELS)
  return tf.image.resizeBilinear(imageTensorGreyscale, [GRID_IMAGE_HEIGHT, GRID_IMAGE_WIDTH])
}

const onPredictGrid = async () => {
  const urls = R.pluck('url', testData)
  const promises = urls.map(loadImage)
  const imageTensors = await Promise.all(promises)
  const input = tf.stack(imageTensors)
  const output = models.grid.model.predict(input)
  const cornersTargetsArray = testData.map(item => calculateCorners(item.boundingBox))
  const cornersPredictionsArray = output.arraySync()
  imageTensors.map(async (imageTensor, index) => {
    const cornersTarget = cornersTargetsArray[index]
    const cornersPrediction = cornersPredictionsArray[index]
    console.log(`cornersTarget [${index}]: ${JSON.stringify(cornersTarget)}`)
    console.log(`cornersPrediction [${index}]: ${JSON.stringify(cornersPrediction)}`)
    const parentElement = document.querySelector('body')
    const canvas = await drawGridImageTensor(parentElement, imageTensor)
    drawCorners(canvas, cornersTarget, 'blue')
    drawCorners(canvas, cornersPrediction, 'red')
  })
}

const BLANK_PREDICTION_ACCURACY = 0.25
const BLANK_PREDICTION_LOWER_LIMIT = 1 - BLANK_PREDICTION_ACCURACY
const DIGIT_PREDICTION_UPPER_LIMIT = 0 + BLANK_PREDICTION_ACCURACY

const isBlankPredictionTooInaccurate = p =>
  p > DIGIT_PREDICTION_UPPER_LIMIT && p < BLANK_PREDICTION_LOWER_LIMIT

const isBlankPredictionCorrect = (label, prediction) =>
  (label && prediction >= BLANK_PREDICTION_LOWER_LIMIT) ||
  (!label && prediction <= DIGIT_PREDICTION_UPPER_LIMIT)

// convincingly blank => 1
// convincingly digit => 0
// anything else => -1
const normaliseBlankPrediction = prediction =>
  prediction >= BLANK_PREDICTION_LOWER_LIMIT
    ? 1
    : (prediction <= DIGIT_PREDICTION_UPPER_LIMIT ? 0 : -1)

const onPredictBlanks = async () => {

  const parentElement = document.getElementById('blanksPredictions')
  deleteChildren(parentElement)

  const yss = []
  const predictionsArrays = []

  const groups = await loadGridSquaresGrouped(testData)
  for (const group of groups) {
    const { xs, ys, gridImageTensor, gridSquaresWithDetails } = group
    const labelsArray = ys.arraySync()
    const outputs = models.blanks.model.predict(xs)
    const predictionsArray = outputs.arraySync()
    const canvas = await drawGridImageTensor(parentElement, gridImageTensor)
    const ctx = canvas.getContext('2d')
    for (const { index, gridSquare } of gridSquaresWithDetails) {
      const label = labelsArray[index]
      const prediction = predictionsArray[index]
      const colour = isBlankPredictionTooInaccurate(prediction)
        ? 'orange'
        : isBlankPredictionCorrect(label, prediction) ? 'green' : 'red'
      ctx.strokeStyle = colour
      ctx.strokeRect(...gridSquare)
    }
    yss.push(ys)
    predictionsArrays.push(predictionsArray)
  }

  const overallLabels = tf.concat(yss)
  const overallPredictionsArray = R.unnest(predictionsArrays)
  const overallPredictions = tf.tensor1d(overallPredictionsArray.map(normaliseBlankPrediction))
  const classAccuracy = await tfvis.metrics.perClassAccuracy(overallLabels, overallPredictions)
  const container = { name: 'Accuracy', tab: 'Evaluation' }
  const classNames = ['Blank', 'Digit']
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames)
}

const onPredictDigits = async () => {

  const parentElement = document.getElementById('digitsPredictions')
  deleteChildren(parentElement)

  const yss = []
  const outputsArray = []

  const groups = await loadDigitsGrouped(testData)
  for (const group of groups) {
    const { xs, ys, gridImageTensor, digitsAndGridSquares } = group
    const labelsArray = ys.argMax(1).arraySync()
    const outputs = models.digits.model.predict(xs)
    const predictionsArray = outputs.argMax(1).arraySync()
    const canvas = await drawGridImageTensor(parentElement, gridImageTensor)
    const ctx = canvas.getContext('2d')
    for (const { index, gridSquare } of digitsAndGridSquares) {
      const correct = predictionsArray[index] === labelsArray[index]
      ctx.strokeStyle = correct ? 'green' : 'red'
      ctx.strokeRect(...gridSquare)
    }
    yss.push(ys)
    outputsArray.push(outputs)
  }

  const overallLabels = tf.concat(yss).argMax(1)
  const overallPredictions = tf.concat(outputsArray).argMax(1)
  const classAccuracy = await tfvis.metrics.perClassAccuracy(overallLabels, overallPredictions)
  const container = { name: 'Accuracy', tab: 'Evaluation' }
  const classNames = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames)
}

const toRows = indexedDigitPredictions =>
  indexedDigitPredictions.map(({ digitPrediction, index }) => ({
    coords: {
      row: Math.trunc(index / 9),
      col: index % 9
    },
    isInitialValue: true,
    value: digitPrediction
  }))

// Using given bounding boxes, predict blanks then predict digits
// Draw resultant images highlighting correct/incorrect predictions
const onPredictBlanksDigits = async () => {
  const data = await loadGridSquaresGrouped(testData)
  const parentElement = document.getElementById('blanksDigitsPredictions')
  deleteChildren(parentElement)
  for (const datum of data) {

    const { xs, gridImageTensor, gridSquaresWithDetails } = datum

    parentElement.appendChild(document.createElement('br'))
    const canvas = await drawGridImageTensor(parentElement, gridImageTensor)
    const ctx = canvas.getContext('2d')

    const blanksPredictionsArray = models.blanks.model.predict(xs).arraySync()

    if (blanksPredictionsArray.some(isBlankPredictionTooInaccurate)) {
      drawBigRedCross(ctx, datum.item.boundingBox)
      continue
    }

    const [blanks, digits] = R.partition(({ index }) =>
      blanksPredictionsArray[index] >= BLANK_PREDICTION_LOWER_LIMIT, gridSquaresWithDetails)

    for (const { isBlank, gridSquare } of blanks) {
      ctx.strokeStyle = isBlank ? 'green' : 'red'
      ctx.lineWidth = 1
      ctx.strokeRect(...gridSquare)
    }

    const xsarr = tf.unstack(xs)
    const indexedDigitPredictions = digits.map(({ index, digit, gridSquare }) => {
      const x = xsarr[index]
      const inputs = tf.stack([x])
      const outputs = models.digits.model.predict(inputs)
      const digitPrediction = outputs.argMax(1).arraySync()[0] + 1
      ctx.strokeStyle = digitPrediction === digit ? 'green' : 'red'
      ctx.lineWidth = 1
      ctx.strokeRect(...gridSquare)
      return { digitPrediction, index }
    })

    const rows = toRows(indexedDigitPredictions)
    const svgElement = createSvgElement('svg', { 'class': 'sudoku-grid' })
    parentElement.appendChild(svgElement)
    drawInitialGrid(svgElement, rows)
  }
}

// Predict bounding boxes then predict blanks then predict digits
// Draw resultant images highlighting correct/incorrect predictions
const onPredictGridBlanksDigits = async () => {
  // TODO:
  // - use model.grid.model to predict the bounding boxes
  // - then, essentially do onPredictBlanksDigits
}

// Predict bounding boxes then predict blanks then predict digits
// Draw resultant images highlighting correct/incorrect predictions
// NOTE: this function processes a single image captured from the webcam (as opposed to testData)
const onPredictCapture = async () => {
  // TODO:
  // - normalise imageData
  // - then, essentially do onPredictGridBlanksDigits but for a single imageTensor

  // const imageTensor = normaliseGridImage(imageData)
  // const input = tf.stack([imageTensor])
  // const output = model.predict(input)
  // const boundingBoxPrediction = output.arraySync()[0]
  // console.log(`boundingBoxPrediction: ${JSON.stringify(boundingBoxPrediction)}`)
  // drawImageTensor(imageTensor, undefined, boundingBoxPrediction)
  // const body = document.querySelector('body')
  // drawGridImageTensor(body, imageTensor)
}

const onSaveModel = name => async () => {
  try {
    console.log(`Saving model ${name}...`)
    const saveResult = await models[name].save(`${location.origin}/api/saveModel/${name}`)
    console.dir(saveResult)
  } catch (error) {
    console.log(`[onSaveModel(${name})] ERROR: ${error.message}`)
  }
}

const onLoadModel = name => async () => {
  try {
    console.log(`Loading model ${name}...`)
    models[name].model = await tf.loadLayersModel(`${location.origin}/models/${name}/model.json`)
    models[name].trained = true
    updateButtonStates()
  } catch (error) {
    console.log(`[onLoadModel(${name})] ERROR: ${error.message}`)
  }
}

const updateButtonStates = () => {

  saveGridBtn.disabled = !models.grid.trained
  saveBlanksBtn.disabled = !models.blanks.trained
  saveDigitsBtn.disabled = !models.digits.trained

  // const allTrained = models.grid.trained && models.blanks.trained && models.digits.trained

  predictGridBtn.disabled = !models.grid.trained
  predictBlanksBtn.disabled = !models.blanks.trained
  predictDigitsBtn.disabled = !models.digits.trained
  predictBlanksDigitsBtn.disabled = !(models.blanks.trained && models.digits.trained)
  predictGridBlanksDigitsBtn.disabled = true // !allTrained
  predictCaptureBtn.disabled = true // !(allTrained && imageData)
}

// Grid

const trainGridBtn = document.getElementById('trainGridBtn')
trainGridBtn.addEventListener('click', onTrainGrid)

const saveGridBtn = document.getElementById('saveGridBtn')
saveGridBtn.addEventListener('click', onSaveModel('grid'))

const loadGridBtn = document.getElementById('loadGridBtn')
loadGridBtn.addEventListener('click', onLoadModel('grid'))

// Blanks

const trainBlanksBtn = document.getElementById('trainBlanksBtn')
trainBlanksBtn.addEventListener('click', onTrainBlanks)

const saveBlanksBtn = document.getElementById('saveBlanksBtn')
saveBlanksBtn.addEventListener('click', onSaveModel('blanks'))

const loadBlanksBtn = document.getElementById('loadBlanksBtn')
loadBlanksBtn.addEventListener('click', onLoadModel('blanks'))

// Digits

const trainDigitsBtn = document.getElementById('trainDigitsBtn')
trainDigitsBtn.addEventListener('click', onTrainDigits)

const saveDigitsBtn = document.getElementById('saveDigitsBtn')
saveDigitsBtn.addEventListener('click', onSaveModel('digits'))

const loadDigitsBtn = document.getElementById('loadDigitsBtn')
loadDigitsBtn.addEventListener('click', onLoadModel('digits'))

// Predictions

const predictGridBtn = document.getElementById('predictGridBtn')
predictGridBtn.addEventListener('click', onPredictGrid)

const clearGridPredictionsBtn = document.getElementById('clearGridPredictionsBtn')
clearGridPredictionsBtn.addEventListener('click', () => deleteChildren(document.getElementById('gridPredictions')))

const predictBlanksBtn = document.getElementById('predictBlanksBtn')
predictBlanksBtn.addEventListener('click', onPredictBlanks)

const clearBlanksPredictionsBtn = document.getElementById('clearBlanksPredictionsBtn')
clearBlanksPredictionsBtn.addEventListener('click', () => deleteChildren(document.getElementById('blanksPredictions')))

const predictDigitsBtn = document.getElementById('predictDigitsBtn')
predictDigitsBtn.addEventListener('click', onPredictDigits)

const clearDigitsPredictionsBtn = document.getElementById('clearDigitsPredictionsBtn')
clearDigitsPredictionsBtn.addEventListener('click', () => deleteChildren(document.getElementById('digitsPredictions')))

const predictBlanksDigitsBtn = document.getElementById('predictBlanksDigitsBtn')
predictBlanksDigitsBtn.addEventListener('click', onPredictBlanksDigits)

const clearBlanksDigitsPredictionsBtn = document.getElementById('clearBlanksDigitsPredictionsBtn')
clearBlanksDigitsPredictionsBtn.addEventListener('click', () => deleteChildren(document.getElementById('blanksDigitsPredictions')))

const predictGridBlanksDigitsBtn = document.getElementById('predictGridBlanksDigitsBtn')
predictGridBlanksDigitsBtn.addEventListener('click', onPredictGridBlanksDigits)

const clearGridBlanksDigitsPredictionsBtn = document.getElementById('clearGridBlanksDigitsPredictionsBtn')
clearGridBlanksDigitsPredictionsBtn.addEventListener('click', () => deleteChildren(document.getElementById('gridBlanksDigitsPredictions')))

const predictCaptureBtn = document.getElementById('predictCaptureBtn')
predictCaptureBtn.addEventListener('click', onPredictCapture)

updateButtonStates()

const main = async () => {
  drawGuides()
  initialiseCamera()
}

main()
