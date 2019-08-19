import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import * as R from 'ramda'
import axios from 'axios'

import * as C from './constants'
import * as D from './data'
import * as I from './image'
// import * as CALC from './calculations'
import * as DC from './drawCanvas'
import * as DS from './drawSvg'
import * as SC from './simpleComponents'
import * as U from './utils'

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
let visor = undefined

const getVisor = () => {
  if (!visor) {
    visor = tfvis.visor()
  }
  visor.open()
  return visor
}

const getTrainingElement = (name, selector) =>
  document.querySelector(`#training-section-${name} ${selector}`)

const getPredictionElement = (name, selector) =>
  document.querySelector(`#prediction-section-${name} ${selector}`)

const createGridModel = () => {

  const inputShape = [C.GRID_IMAGE_HEIGHT, C.GRID_IMAGE_WIDTH, C.GRID_IMAGE_CHANNELS]
  // const conv2dArgs = {
  //   kernelSize: 7,
  //   filters: 8,
  //   // activation: 'sigmoid',
  //   activation: 'tanh',
  //   strides: 1,
  //   kernelInitializer: 'varianceScaling'
  // }
  const maxPooling2dArgs = {
    poolSize: [2, 2],
    strides: [2, 2]
  }

  const model = tf.sequential()

  model.add(tf.layers.conv2d({ inputShape, kernelSize: 3, filters: 8, strides: 1, activation: 'relu6' }))
  model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 8, strides: 2, activation: 'relu6' }))
  model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  model.add(tf.layers.conv2d({ kernelSize: 1, filters: 8, strides: 1 }))
  // model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  // model.add(tf.layers.conv2d({ ...conv2dArgs, activation: 'tanh' }))
  // model.add(tf.layers.maxPooling2d(maxPooling2dArgs))
  // model.add(tf.layers.conv2d({ ...conv2dArgs, activation: 'tanh' }))

  model.add(tf.layers.flatten())

  model.add(tf.layers.dense({ units: 200, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 4 }))

  model.summary()

  return model
}

const createBlanksModel = () => {
  const inputShape = [C.DIGIT_IMAGE_HEIGHT, C.DIGIT_IMAGE_WIDTH, C.DIGIT_IMAGE_CHANNELS]
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

  const inputShape = [C.DIGIT_IMAGE_HEIGHT, C.DIGIT_IMAGE_WIDTH, C.DIGIT_IMAGE_CHANNELS]
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
  const parentElement = getTrainingElement('grid', '.training-data')
  const { xs, ys } = await D.loadGridData(combinedData, parentElement)

  model.compile({
    optimizer: 'rmsprop',
    // loss: 'meanSquaredError'
    loss: 'meanAbsoluteError'
  })

  const trainingSurface = getVisor().surface({ tab: 'Grid', name: 'Model Training' })
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
    validationSplit: 0.20,
    callbacks: customCallback
  }

  return model.fit(xs, ys, params)
}

const trainBlanks = async model => {

  const combinedData = trainingData.concat(validationData)
  tf.util.shuffle(combinedData)
  const { xs, ys } = await D.loadGridSquaresFlat(combinedData)

  model.compile({
    optimizer: 'rmsprop',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  })

  const trainingSurface = getVisor().surface({ tab: 'Blanks', name: 'Model Training' })
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
  const { xs, ys } = await D.loadDigitsFlat(combinedData)

  model.compile({
    optimizer: 'rmsprop',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  const trainingSurface = getVisor().surface({ tab: 'Digits', name: 'Model Training' })
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
  DS.createSvgElement('path', { d, class: 'video-guide' })

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
  const startBtn = document.getElementById('start-btn')
  const stopBtn = document.getElementById('stop-btn')
  const captureBtn = document.getElementById('capture-btn')
  const saveBtn = document.getElementById('save-btn')
  const clearBtn = document.getElementById('clear-btn')
  const messageArea = document.getElementById('message-area')

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
    canvas.width = C.GRID_IMAGE_WIDTH
    canvas.height = C.GRID_IMAGE_HEIGHT
    const imageTensor = I.normaliseGridImage(imageData)
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
  const trainGridBtn = getTrainingElement('grid', '.train-btn')
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
  const trainBlanksBtn = getTrainingElement('blanks', '.train-btn')
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
  const trainDigitsBtn = getTrainingElement('digits', '.train-btn')
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

const onPredictGrid = async () => {
  const urls = R.pluck('url', testData)
  const promises = urls.map(D.loadImage)
  const imageTensors = await Promise.all(promises)
  const input = tf.stack(imageTensors)
  const output = models.grid.model.predict(input)
  const targetsArray = testData.map(item => item.boundingBox)
  const predictionsArray = output.arraySync()
  const parentElement = getPredictionElement('grid', '.results')
  U.deleteChildren(parentElement)
  const promises2 = imageTensors.map(async (imageTensor, index) => {
    const target = targetsArray[index]
    const prediction = predictionsArray[index]
    console.log(`target [${index}]: ${JSON.stringify(target)}`)
    console.log(`prediction [${index}]: ${JSON.stringify(prediction)}`)
    const canvas = await DC.drawGridImageTensor(parentElement, imageTensor)
    DC.drawBoundingBox(canvas, target, 'blue')
    DC.drawBoundingBox(canvas, prediction, 'red')
  })
  await Promise.all(promises2)
  updateButtonStates()
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

  const parentElement = getPredictionElement('blanks', '.results')
  U.deleteChildren(parentElement)

  const yss = []
  const predictionsArrays = []

  const groups = await D.loadGridSquaresGrouped(testData)
  for (const group of groups) {
    const { xs, ys, gridImageTensor, gridSquaresWithDetails } = group
    const labelsArray = ys.arraySync()
    const outputs = models.blanks.model.predict(xs)
    const predictionsArray = outputs.arraySync()
    const canvas = await DC.drawGridImageTensor(parentElement, gridImageTensor)
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
  const surface = getVisor().surface({ name: 'Accuracy', tab: 'Evaluation' })
  const classNames = ['Blank', 'Digit']
  tfvis.show.perClassAccuracy(surface, classAccuracy, classNames)

  updateButtonStates()
}

const onPredictDigits = async () => {

  const parentElement = getPredictionElement('digits', '.results')
  U.deleteChildren(parentElement)

  const yss = []
  const outputsArray = []

  const groups = await D.loadDigitsGrouped(testData)
  for (const group of groups) {
    const { xs, ys, gridImageTensor, gridSquaresWithDetails } = group
    const labelsArray = ys.argMax(1).arraySync()
    const outputs = models.digits.model.predict(xs)
    const predictionsArray = outputs.argMax(1).arraySync()
    const canvas = await DC.drawGridImageTensor(parentElement, gridImageTensor)
    const ctx = canvas.getContext('2d')
    for (const { index, gridSquare } of gridSquaresWithDetails) {
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
  const surface = getVisor().surface({ name: 'Accuracy', tab: 'Evaluation' })
  const classNames = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
  tfvis.show.perClassAccuracy(surface, classAccuracy, classNames)

  updateButtonStates()
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
  const data = await D.loadGridSquaresGrouped(testData)
  const parentElement = getPredictionElement('blanks-digits', '.results')
  U.deleteChildren(parentElement)
  for (const datum of data) {

    const { xs, gridImageTensor, gridSquaresWithDetails } = datum

    parentElement.appendChild(document.createElement('br'))
    const canvas = await DC.drawGridImageTensor(parentElement, gridImageTensor)
    const ctx = canvas.getContext('2d')

    const blanksPredictionsArray = models.blanks.model.predict(xs).arraySync()

    if (blanksPredictionsArray.some(isBlankPredictionTooInaccurate)) {
      DC.drawBigRedCross(ctx, datum.item.boundingBox)
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
    const svgElement = DS.createSvgElement('svg', { 'class': 'sudoku-grid' })
    parentElement.appendChild(svgElement)
    DS.drawInitialGrid(svgElement, rows)
  }

  updateButtonStates()
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

  // const imageTensor = I.normaliseGridImage(imageData)
  // const input = tf.stack([imageTensor])
  // const output = model.predict(input)
  // const boundingBoxPrediction = output.arraySync()[0]
  // console.log(`boundingBoxPrediction: ${JSON.stringify(boundingBoxPrediction)}`)
  // drawImageTensor(imageTensor, undefined, boundingBoxPrediction)
  // const body = document.querySelector('body')
  // DC.drawGridImageTensor(body, imageTensor)
}

const onSaveModel = name => async () => {
  try {
    SC.hideErrorPanel()
    console.log(`Saving model ${name}...`)
    const saveResult = await models[name].model.save(`${location.origin}/api/saveModel/${name}`)
    console.dir(saveResult)
  } catch (error) {
    console.log(`[onSaveModel(${name})] ERROR: ${error.message}`)
    SC.showErrorPanel(error.message)
  }
}

const onLoadModel = name => async () => {
  try {
    SC.hideErrorPanel()
    console.log(`Loading model ${name}...`)
    models[name].model = await tf.loadLayersModel(`${location.origin}/models/${name}/model.json`)
    models[name].trained = true
    updateButtonStates()
  } catch (error) {
    console.log(`[onLoadModel(${name})] ERROR: ${error.message}`)
    SC.showErrorPanel(error.message)
  }
}

const updateButtonStates = () => {

  const saveGridBtn = getTrainingElement('grid', '.save-btn')
  const saveBlanksBtn = getTrainingElement('blanks', '.save-btn')
  const saveDigitsBtn = getTrainingElement('digits', '.save-btn')

  saveGridBtn.disabled = !models.grid.trained
  saveBlanksBtn.disabled = !models.blanks.trained
  saveDigitsBtn.disabled = !models.digits.trained

  const allTrained = false && models.grid.trained && models.blanks.trained && models.digits.trained

  const predictGridBtn = getPredictionElement('grid', '.predict-btn')
  const predictBlanksBtn = getPredictionElement('blanks', '.predict-btn')
  const predictDigitsBtn = getPredictionElement('digits', '.predict-btn')
  const predictBlanksDigitsBtn = getPredictionElement('blanks-digits', '.predict-btn')
  const predictGridBlanksDigitsBtn = getPredictionElement('grid-blanks-digits', '.predict-btn')

  predictGridBtn.disabled = !models.grid.trained
  predictBlanksBtn.disabled = !models.blanks.trained
  predictDigitsBtn.disabled = !models.digits.trained
  predictBlanksDigitsBtn.disabled = !(models.blanks.trained && models.digits.trained)
  predictGridBlanksDigitsBtn.disabled = !allTrained
  predictCaptureBtn.disabled = !(allTrained && imageData)

  const clearGridBtn = getPredictionElement('grid', '.clear-btn')
  const clearBlanksBtn = getPredictionElement('blanks', '.clear-btn')
  const clearDigitsBtn = getPredictionElement('digits', '.clear-btn')
  const clearBlanksDigitsBtn = getPredictionElement('blanks-digits', '.clear-btn')
  const clearGridBlanksDigitsBtn = getPredictionElement('grid-blanks-digits', '.clear-btn')

  const gridResults = getPredictionElement('grid', '.results')
  const blanksResults = getPredictionElement('blanks', '.results')
  const digitsResults = getPredictionElement('digits', '.results')
  const blanksDigitsResults = getPredictionElement('blanks-digits', '.results')
  const gridBlanksDigitsResults = getPredictionElement('grid-blanks-digits', '.results')

  clearGridBtn.disabled = !gridResults.hasChildNodes()
  clearBlanksBtn.disabled = !blanksResults.hasChildNodes()
  clearDigitsBtn.disabled = !digitsResults.hasChildNodes()
  clearBlanksDigitsBtn.disabled = !blanksDigitsResults.hasChildNodes()
  clearGridBlanksDigitsBtn.disabled = !gridBlanksDigitsResults.hasChildNodes()

  showVisorBtn.disabled = !visor
}

SC.addTrainingSection('grid', onTrainGrid, onSaveModel, onLoadModel)
SC.addTrainingSection('blanks', onTrainBlanks, onSaveModel, onLoadModel)
SC.addTrainingSection('digits', onTrainDigits, onSaveModel, onLoadModel)

SC.addPredictionSection('grid', onPredictGrid)
SC.addPredictionSection('blanks', onPredictBlanks)
SC.addPredictionSection('digits', onPredictDigits)
SC.addPredictionSection('blanks-digits', onPredictBlanksDigits)
SC.addPredictionSection('grid-blanks-digits', onPredictGridBlanksDigits)

document.querySelectorAll('.clear-btn').forEach(clearBtn =>
  clearBtn.addEventListener('click', () => {
    visor && visor.close()
    updateButtonStates()
  }))

const predictCaptureBtn = document.getElementById('predict-capture-btn')
predictCaptureBtn.addEventListener('click', onPredictCapture)

const showVisorBtn = document.getElementById('show-visor-btn')
showVisorBtn.addEventListener('click', () => visor && visor.open())

updateButtonStates()

const main = async () => {
  drawGuides()
  initialiseCamera()
}

main()
