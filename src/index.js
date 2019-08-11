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

let gridModel = undefined
let blanksModel = undefined
let digitsModel = undefined

let trainedGrid = false
let trainedBlanks = false
let trainedDigits = false

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

const drawGridSquares = (ctx, boundingBox, colour) => {
  for (const gridSquare of calculateGridSquares(boundingBox)) {
    ctx.strokeStyle = colour
    ctx.strokeRect(...gridSquare)
  }
}

// const drawMajorInnerGridLines = (ctx, coordsList, colour) => {
//   ctx.beginPath()
//   R.range(0, 4).forEach(idx => {
//     const [x1, y1, x2, y2] = coordsList.slice(idx * 4)
//     ctx.moveTo(x1, y1)
//     ctx.lineTo(x2, y2)
//   })
//   ctx.strokeStyle = colour
//   ctx.lineWidth = 1
//   ctx.stroke()
// }

const drawGridImageTensor = async (parentElement, imageTensor, boundingBoxTarget, boundingBoxPrediction) => {
  const canvas = document.createElement('canvas')
  canvas.setAttribute('class', 'grid-image')
  await tf.browser.toPixels(imageTensor, canvas)
  const ctx = canvas.getContext('2d')
  if (boundingBoxTarget) {
    ctx.strokeStyle = 'blue'
    ctx.strokeRect(...boundingBoxTarget)
  }
  if (boundingBoxPrediction) {
    ctx.strokeStyle = 'red'
    ctx.strokeRect(...boundingBoxPrediction)
    drawGridSquares(ctx, boundingBoxPrediction, 'red')
  }
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

const loadGridData = async data => {
  const urls = R.pluck('url', data)
  const promises = urls.map(loadImage)
  const imageTensors = await Promise.all(promises)
  const boundingBoxes = R.pluck('boundingBox', data)
  const xs = tf.stack(imageTensors)
  const ys = tf.tensor2d(boundingBoxes, undefined, 'int32')
  return { xs, ys }
}

// Probably need to use tf.tidy somewhere in here ?
const cropDigitImagesFromGridImage = (item, gridImageTensor) => {
  const { puzzleId, boundingBox, } = item
  const gridSquares = Array.from(calculateGridSquares(boundingBox))
  const puzzle = puzzles.find(p => p.id === puzzleId)
  const flattenedInitialValues = Array.from(puzzle.initialValues.join(''))
  const digitsAndGridSquares = flattenedInitialValues
    .map((ch, index) => ({ digit: Number(ch), index, gridSquare: gridSquares[index] }))
    .filter(({ digit }) => Number.isInteger(digit) && digit >= 1 && digit <= 9)
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

// Probably need to use tf.tidy somewhere in here ?
const cropAllGridSquareImagesFromGridImage = (item, gridImageTensor) => {
  const { puzzleId, boundingBox, } = item
  const gridSquares = Array.from(calculateGridSquares(boundingBox))
  const puzzle = puzzles.find(p => p.id === puzzleId)
  const flattenedInitialValues = Array.from(puzzle.initialValues.join(''))
  const gridSquaresWithDetails = flattenedInitialValues
    .map((ch, index) => ({
      isBlank: ch === ' ',
      digit: Number(ch),
      index,
      gridSquare: gridSquares[index]
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

// Probably need to use tf.tidy somewhere in here ?
const loadAllGridSquaresData = async data => {
  const urls = R.pluck('url', data)
  const promises = urls.map(loadImage)
  const gridImageTensors = await Promise.all(promises)
  const perGrid = gridImageTensors.map((gridImageTensor, index) => {
    const item = data[index]
    return cropAllGridSquareImagesFromGridImage(item, gridImageTensor)
  })
  const xss = R.pluck('xs', perGrid)
  const yss = R.pluck('ys', perGrid)
  const xs = tf.concat(xss)
  const ys = tf.concat(yss)
  return { xs, ys }
}

// Probably need to use tf.tidy somewhere in here ?
const loadAllGridSquaresData2 = async data => {
  const urls = R.pluck('url', data)
  const promises = urls.map(loadImage)
  const gridImageTensors = await Promise.all(promises)
  return gridImageTensors.map((gridImageTensor, index) => {
    const item = data[index]
    return cropAllGridSquareImagesFromGridImage(item, gridImageTensor)
  })
}

// Probably need to use tf.tidy somewhere in here ?
const loadDigitData = async data => {
  const urls = R.pluck('url', data)
  const promises = urls.map(loadImage)
  const gridImageTensors = await Promise.all(promises)
  const perGrid = gridImageTensors.map((gridImageTensor, index) => {
    const item = data[index]
    return cropDigitImagesFromGridImage(item, gridImageTensor)
  })
  const xss = R.pluck('xs', perGrid)
  const yss = R.pluck('ys', perGrid)
  const xs = tf.concat(xss)
  const ys = tf.concat(yss)
  return { xs, ys }
}

// Probably need to use tf.tidy somewhere in here ?
const loadDigitData2 = async data => {
  const urls = R.pluck('url', data)
  const promises = urls.map(loadImage)
  const gridImageTensors = await Promise.all(promises)
  return gridImageTensors.map((gridImageTensor, index) => {
    const item = data[index]
    return cropDigitImagesFromGridImage(item, gridImageTensor)
  })
}

const createGridModel = () => {

  const inputShape = [GRID_IMAGE_HEIGHT, GRID_IMAGE_WIDTH, GRID_IMAGE_CHANNELS]
  const conv2dArgs = {
    kernelSize: 3,
    filters: 32,
    activation: 'sigmoid',
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
  model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  model.add(tf.layers.conv2d(conv2dArgs))
  model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  model.add(tf.layers.conv2d(conv2dArgs))
  model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  model.add(tf.layers.conv2d(conv2dArgs))

  model.add(tf.layers.flatten())

  model.add(tf.layers.dense({ units: 100, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 4 }))

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
  const { xs, ys } = await loadGridData(combinedData)

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
    epochs: 20,
    shuffle: true,
    validationSplit: 0.15,
    callbacks: customCallback
  }

  return model.fit(xs, ys, params)
}

const trainBlanks = async model => {

  const combinedData = trainingData.concat(validationData)
  tf.util.shuffle(combinedData)
  const { xs, ys } = await loadAllGridSquaresData(combinedData)

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
  const { xs, ys } = await loadDigitData(combinedData)

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
    predictCaptureBtn.disabled = !(imageData && trainedGrid && trainedBlanks && trainedDigits)
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
    gridModel = createGridModel()
    const trainingResults = await trainGrid(gridModel)
    console.dir(trainingResults)
    const lastLoss = R.last(trainingResults.history.loss)
    const lastValLoss = R.last(trainingResults.history.val_loss)
    console.log('last loss:', lastLoss, 'sqrt:', Math.sqrt(lastLoss))
    console.log('last val_loss:', lastValLoss, 'sqrt:', Math.sqrt(lastValLoss))
    trainedGrid = true
    updateButtonStates()
  } finally {
    trainGridBtn.disabled = false
  }
}

const onTrainBlanks = async () => {
  try {
    trainBlanksBtn.disabled = true
    blanksModel = createBlanksModel()
    await trainBlanks(blanksModel)
    trainedBlanks = true
    updateButtonStates()
  } finally {
    trainBlanksBtn.disabled = false
  }
}

const onTrainDigits = async () => {
  try {
    trainDigitsBtn.disabled = true
    digitsModel = createDigitsModel()
    await trainDigits(digitsModel)
    trainedDigits = true
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

const onPredictCapture = async () => {
  // const imageTensor = normaliseGridImage(imageData)
  // const input = tf.stack([imageTensor])
  // const output = model.predict(input)
  // const boundingBoxPrediction = output.arraySync()[0]
  // console.log(`boundingBoxPrediction: ${JSON.stringify(boundingBoxPrediction)}`)
  // drawImageTensor(imageTensor, undefined, boundingBoxPrediction)
  // const body = document.querySelector('body')
  // drawGridImageTensor(body, imageTensor)
}

const onPredictGridTestData = async () => {
  const promises = testData.map(item => item.url).map(loadImage)
  const imageTensors = await Promise.all(promises)
  const input = tf.stack(imageTensors)
  const output = gridModel.predict(input)
  const boundingBoxPredictions = output.arraySync()
  imageTensors.map((imageTensor, index) => {
    const boundingBoxTarget = testData[index].boundingBox
    const boundingBoxPrediction = boundingBoxPredictions[index]
    console.log(`boundingBoxTarget [${index}]: ${JSON.stringify(boundingBoxTarget)}`)
    console.log(`boundingBoxPrediction [${index}]: ${JSON.stringify(boundingBoxPrediction)}`)
    const body = document.querySelector('body')
    drawGridImageTensor(body, imageTensor, boundingBoxTarget, boundingBoxPrediction)
  })
}

const BLANK_THRESHOLD = 0.75
const DIGIT_THRESHOLD = 0.25

const onPredictBlanksTestData = async () => {
  const { xs, ys } = await loadAllGridSquaresData(testData)
  const labels = ys
  const outputs = blanksModel.predict(xs)
  const predictions = tf.tensor1d(outputs.arraySync().map(p => p >= BLANK_THRESHOLD ? 1 : (p <= DIGIT_THRESHOLD ? 0 : -1)))
  const classNames = ['Blank', 'Digit']
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, predictions)
  const container = { name: 'Accuracy', tab: 'Evaluation' }
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames)
}

const onPredictBlanksTestData2 = async () => {
  const data = await loadAllGridSquaresData2(testData)
  for (const datum of data) {
    const { xs, ys, gridImageTensor, gridSquaresWithDetails } = datum
    const labels = ys.arraySync()
    const outputs = blanksModel.predict(xs)
    const predictions = outputs.arraySync()
    const body = document.querySelector('body')
    const canvas = await drawGridImageTensor(body, gridImageTensor)
    const ctx = canvas.getContext('2d')
    for (const { index, gridSquare } of gridSquaresWithDetails) {
      const label = labels[index]
      const prediction = predictions[index]
      const colour = prediction > DIGIT_THRESHOLD && prediction < BLANK_THRESHOLD
        ? 'orange'
        : (label ? (prediction >= BLANK_THRESHOLD ? 'green' : 'red') : (prediction <= DIGIT_THRESHOLD ? 'green' : 'red'))
      ctx.strokeStyle = colour
      ctx.strokeRect(...gridSquare)
    }
  }
}

const onPredictDigitsTestData = async () => {
  const { xs, ys } = await loadDigitData(testData)
  const labels = ys.argMax(1)
  const outputs = digitsModel.predict(xs)
  const predictions = outputs.argMax(1)
  const classNames = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, predictions)
  const container = { name: 'Accuracy', tab: 'Evaluation' }
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames)
}

const onPredictDigitsTestData2 = async () => {
  const data = await loadDigitData2(testData)
  for (const datum of data) {
    const { xs, ys, gridImageTensor, digitsAndGridSquares } = datum
    const labels = ys.argMax(1).arraySync()
    const outputs = digitsModel.predict(xs)
    const predictions = outputs.argMax(1).arraySync()
    const body = document.querySelector('body')
    const canvas = await drawGridImageTensor(body, gridImageTensor)
    const ctx = canvas.getContext('2d')
    for (const { index, gridSquare } of digitsAndGridSquares) {
      const correct = predictions[index] === labels[index]
      ctx.strokeStyle = correct ? 'green' : 'red'
      ctx.strokeRect(...gridSquare)
    }
  }
}

const onPredictBlanksAndDigitsTestData = async () => {
  const data = await loadAllGridSquaresData2(testData)
  for (const datum of data) {

    const { xs, gridImageTensor, gridSquaresWithDetails } = datum

    const parentElement = document.querySelector('body')
    parentElement.appendChild(document.createElement('br'))
    const canvas = await drawGridImageTensor(parentElement, gridImageTensor)
    const ctx = canvas.getContext('2d')

    const blanksPredictions = blanksModel.predict(xs).arraySync()

    if (blanksPredictions.some(p => p > DIGIT_THRESHOLD && p < BLANK_THRESHOLD)) {
      const [x, y, w, h] = datum.item.boundingBox
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
      continue
    }

    const [blanks, digits] = R.partition(({ index }) =>
      blanksPredictions[index] >= BLANK_THRESHOLD, gridSquaresWithDetails)

    for (const { isBlank, gridSquare } of blanks) {
      ctx.strokeStyle = isBlank ? 'green' : 'red'
      ctx.lineWidth = 1
      ctx.strokeRect(...gridSquare)
    }

    const xsarr = tf.unstack(xs)
    const indexedDigitPredictions = digits.map(({ index, digit, gridSquare }) => {
      const x = xsarr[index]
      const inputs = tf.stack([x])
      const outputs = digitsModel.predict(inputs)
      const digitPrediction = outputs.argMax(1).arraySync()[0] + 1
      ctx.strokeStyle = digitPrediction === digit ? 'green' : 'red'
      ctx.lineWidth = 1
      ctx.strokeRect(...gridSquare)
      return { digitPrediction, index }
    })

    const seed = Array(81).fill(' ')
    const reducer = (acc, { digitPrediction, index }) => R.update(index, digitPrediction, acc)
    const flattenedInitialValues = R.reduce(reducer, seed, indexedDigitPredictions)
    const initialValues = R.splitEvery(9, flattenedInitialValues).map(R.join(''))
    console.log(`Puzzle ${datum.item.puzzleId}; URL: ${datum.item.url}`)
    initialValues.forEach(line => console.log(line))

    const svgElement = createSvgElement('svg', { 'class': 'sudoku-grid' })
    parentElement.appendChild(svgElement)
    const rows = indexedDigitPredictions.map(({ digitPrediction, index }) => ({
      coords: {
        row: Math.trunc(index / 9),
        col: index % 9
      },
      isInitialValue: true,
      value: digitPrediction
    }))
    drawInitialGrid(svgElement, rows)
  }
}

const onSaveBlanksModel = async () => {
  try {
    const saveResult = await blanksModel.save(`${location.origin}/api/saveModel/blanks`)
    console.dir(saveResult)
  } catch (error) {
    console.log(`[onSaveBlanks] ERROR: ${error.message}`)
  }
}

const onLoadBlanksModel = async () => {
  try {
    blanksModel = await tf.loadLayersModel(`${location.origin}/models/blanks/model.json`)
    trainedBlanks = true
    updateButtonStates()
  } catch (error) {
    console.log(`[onLoadBlanks] ERROR: ${error.message}`)
  }
}

const onSaveDigitsModel = async () => {
  try {
    const saveResult = await digitsModel.save(`${location.origin}/api/saveModel/digits`)
    console.dir(saveResult)
  } catch (error) {
    console.log(`[onSaveDigitsModel] ERROR: ${error.message}`)
  }
}

const onLoadDigitsModel = async () => {
  try {
    digitsModel = await tf.loadLayersModel(`${location.origin}/models/digits/model.json`)
    trainedDigits = true
    updateButtonStates()
  } catch (error) {
    console.log(`[onLoadDigitsModel] ERROR: ${error.message}`)
  }
}

const updateButtonStates = () => {
  predictGridTestDataBtn.disabled = !trainedGrid
  predictBlanksTestDataBtn.disabled = !trainedBlanks
  predictDigitsTestDataBtn.disabled = !trainedDigits
  predictBlanksAndDigitsTestDataBtn.disabled = !(trainedBlanks && trainedDigits)
  predictCaptureBtn.disabled = true
  saveBlanksBtn.disabled = !trainedBlanks
}

const trainGridBtn = document.getElementById('trainGridBtn')
trainGridBtn.addEventListener('click', onTrainGrid)

const trainBlanksBtn = document.getElementById('trainBlanksBtn')
trainBlanksBtn.addEventListener('click', onTrainBlanks)

const saveBlanksBtn = document.getElementById('saveBlanksBtn')
saveBlanksBtn.addEventListener('click', onSaveBlanksModel)

const loadBlanksBtn = document.getElementById('loadBlanksBtn')
loadBlanksBtn.addEventListener('click', onLoadBlanksModel)

const trainDigitsBtn = document.getElementById('trainDigitsBtn')
trainDigitsBtn.addEventListener('click', onTrainDigits)

const saveDigitsBtn = document.getElementById('saveDigitsBtn')
saveDigitsBtn.addEventListener('click', onSaveDigitsModel)

const loadDigitsBtn = document.getElementById('loadDigitsBtn')
loadDigitsBtn.addEventListener('click', onLoadDigitsModel)

const predictGridTestDataBtn = document.getElementById('predictGridTestDataBtn')
predictGridTestDataBtn.addEventListener('click', onPredictGridTestData)

const predictBlanksTestDataBtn = document.getElementById('predictBlanksTestDataBtn')
predictBlanksTestDataBtn.addEventListener('click', onPredictBlanksTestData)
predictBlanksTestDataBtn.addEventListener('click', onPredictBlanksTestData2)

const predictDigitsTestDataBtn = document.getElementById('predictDigitsTestDataBtn')
predictDigitsTestDataBtn.addEventListener('click', onPredictDigitsTestData)
predictDigitsTestDataBtn.addEventListener('click', onPredictDigitsTestData2)

const predictBlanksAndDigitsTestDataBtn = document.getElementById('predictBlanksAndDigitsTestDataBtn')
predictBlanksAndDigitsTestDataBtn.addEventListener('click', onPredictBlanksAndDigitsTestData)

const predictCaptureBtn = document.getElementById('predictCaptureBtn')
predictCaptureBtn.addEventListener('click', onPredictCapture)

updateButtonStates()

const main = async () => {
  drawGuides()
  initialiseCamera()
}

main()
