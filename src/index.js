import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import * as R from 'ramda'
import * as log from 'loglevel'
import axios from 'axios'

import * as C from './constants'
import * as D from './data'
import * as I from './image'
import * as P from './puzzle'
import * as U from './utils'
import * as DC from './drawCanvas'
import * as CALC from './/calculations'
import * as DS from './drawSvg'
import * as SC from './simpleComponents'
import { solve } from '../utils/solving'

import trainingData from '../data/training-data.json'
import validationData from '../data/validation-data.json'
import testData from '../data/test-data.json'

log.setLevel('info')

const models = {
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

// http://emaraic.com/blog/realtime-sudoku-solver
const findBoundingBox = async gridImageTensor => {

  const tfCanvas = document.createElement('canvas')
  await tf.browser.toPixels(gridImageTensor, tfCanvas)
  const matInitial = cv.imread(tfCanvas)

  const matGrey = new cv.Mat(matInitial.size(), cv.CV_8UC1)
  cv.cvtColor(matInitial, matGrey, cv.COLOR_BGR2GRAY)

  const matBlur = new cv.Mat(matInitial.size(), cv.CV_8UC1)
  const ksize = new cv.Size(5, 5)
  const sigmaX = 0
  cv.GaussianBlur(matGrey, matBlur, ksize, sigmaX)

  const matBinary = new cv.Mat(matInitial.size(), cv.CV_8UC1)
  cv.adaptiveThreshold(matBlur, matBinary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 19, 3)

  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  const mode = cv.RETR_LIST
  const method = cv.CHAIN_APPROX_SIMPLE
  cv.findContours(matBinary, contours, hierarchy, mode, method)
  const numContours = contours.size()
  const areasAndBoundingRects = R.range(0, numContours).map(index => {
    const contour = contours.get(index)
    const area = cv.contourArea(contour)
    const boundingRect = cv.boundingRect(contour)
    return { area, boundingRect }
  })
  const sorted = R.sort(R.descend(R.prop('area')), areasAndBoundingRects)
  const { x, y, width, height } = R.head(sorted).boundingRect

  // I'm insetting by 2 pixels in both directions because
  // the best contour tends to be just slightly too big.
  const boundingBox = CALC.inset(x, y, width, height, 2, 2)

  return boundingBox
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

const trainBlanks = async model => {

  const combinedData = trainingData.concat(validationData)
  tf.util.shuffle(combinedData)
  const { xs, ys } = await D.loadGridSquaresFromKnownGrids(combinedData)

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
  const { xs, ys } = await D.loadDigitsFromKnownGrids(combinedData)

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

const WEBCAM_MODE_VIDEO = Symbol('WEBCAM_MODE_VIDEO')
const WEBCAM_MODE_CAPTURE = Symbol('WEBCAM_MODE_CAPTURE')
const WEBCAM_MODE_SOLUTION = Symbol('WEBCAM_MODE_SOLUTION')

const setWebcamMode = mode => {
  const videoElement = document.getElementById('video')
  const videoGuidesElement = document.getElementById('video-guides')
  const capturedImageElement = document.getElementById('captured-image')
  const predictCaptureResults = document.getElementById('predict-capture-results')
  videoElement.style.display = mode === WEBCAM_MODE_VIDEO ? 'inline-block' : 'none'
  videoGuidesElement.style.display = mode === WEBCAM_MODE_VIDEO ? 'inline-block' : 'none'
  capturedImageElement.style.display = mode === WEBCAM_MODE_CAPTURE ? 'inline-block' : 'none'
  predictCaptureResults.style.display = mode === WEBCAM_MODE_SOLUTION ? 'inline-block' : 'none'
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
  const predictCaptureResults = document.getElementById('predict-capture-results')

  const updateButtonState = () => {
    const playing = !!videoElement.srcObject
    startBtn.disabled = playing
    stopBtn.disabled = !playing
    captureBtn.disabled = !playing
    saveBtn.disabled = !imageData
    clearBtn.disabled = !imageData
    const allTrained = models.blanks.trained && models.digits.trained
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
      setWebcamMode(WEBCAM_MODE_VIDEO)
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
    setWebcamMode(WEBCAM_MODE_CAPTURE)
  }

  const onSave = async () => {

    SC.hideErrorPanel()

    try {
      const dataUrlRaw = capturedImageElement.toDataURL('image/png')
      log.info(`Saving raw image...`)
      const responseRaw = await axios.post('/api/saveRawImage', { dataUrl: dataUrlRaw })
      messageArea.innerText = responseRaw.data
    } catch (error) {
      log.error(`[onSave] /api/saveRawImage: ${error}`)
      SC.showErrorPanel('Failed to save raw image.')
    }

    try {
      const canvas = document.createElement('canvas')
      canvas.width = C.GRID_IMAGE_WIDTH
      canvas.height = C.GRID_IMAGE_HEIGHT
      const imageTensor = I.normaliseGridImage(imageData)
      await tf.browser.toPixels(imageTensor, canvas)
      const dataUrlNormalised = canvas.toDataURL('image/png')
      log.info(`Saving normalised image...`)
      await axios.post('/api/saveNormalisedImage', { dataUrl: dataUrlNormalised })
    } catch (error) {
      log.error(`[onSave] /api/saveNormalisedImage: ${error}`)
      SC.showErrorPanel('Failed to save normalised image.')
    }
  }

  const onClear = () => {
    capturedImageElementContext.clearRect(0, 0, capturedImageElement.width, capturedImageElement.height)
    U.deleteChildren(predictCaptureResults)
    imageData = undefined
    messageArea.innerText = ''
    updateButtonState()
    setWebcamMode(WEBCAM_MODE_VIDEO)
  }

  startBtn.addEventListener('click', onStart)
  stopBtn.addEventListener('click', onStop)
  captureBtn.addEventListener('click', onCapture)
  saveBtn.addEventListener('click', onSave)
  clearBtn.addEventListener('click', onClear)

  updateButtonState()
  setWebcamMode(WEBCAM_MODE_VIDEO)
}

const onTrainBlanks = async () => {
  const trainBlanksBtn = getTrainingElement('blanks', '.train-btn')
  try {
    SC.hideErrorPanel()
    trainBlanksBtn.disabled = true
    models.blanks.model = createBlanksModel()
    await trainBlanks(models.blanks.model)
    models.blanks.trained = true // eslint-disable-line
  } catch (error) {
    log.error(`[onTrainBlanks] ${error.message}`)
    SC.showErrorPanel(error.message)
  } finally {
    trainBlanksBtn.disabled = false
  }
}

const onTrainDigits = async () => {
  const trainDigitsBtn = getTrainingElement('digits', '.train-btn')
  try {
    SC.hideErrorPanel()
    trainDigitsBtn.disabled = true
    models.digits.model = createDigitsModel()
    await trainDigits(models.digits.model)
    models.digits.trained = true // eslint-disable-line
  } catch (error) {
    log.error(`[onTrainDigits] ${error.message}`)
    SC.showErrorPanel(error.message)
  } finally {
    trainDigitsBtn.disabled = false
  }
}

const onPredictGrid = async () => {
  try {
    SC.hideErrorPanel()
    const parentElement = getPredictionElement('grid', '.results')
    U.deleteChildren(parentElement)
    const promises = testData.map(async item => {
      const gridImageTensor = await I.loadImage(item.url)
      const boundingBox = await findBoundingBox(gridImageTensor)
      const canvas = await DC.drawGridImageTensor(parentElement, gridImageTensor)
      DC.drawBoundingBox(canvas, item.boundingBox, 'blue')
      DC.drawBoundingBox(canvas, boundingBox, 'red')
      return boundingBox
    })
    await Promise.all(promises)
  } catch (error) {
    log.error(`[onPredictGrid] ${error.message}`)
    SC.showErrorPanel(error.message)
  }
}

const BLANK_PREDICTION_ACCURACY = 0.25
const BLANK_PREDICTION_LOWER_LIMIT = 1 - BLANK_PREDICTION_ACCURACY
const DIGIT_PREDICTION_UPPER_LIMIT = 0 + BLANK_PREDICTION_ACCURACY

const isBlankPredictionTooInaccurate = p =>
  p > DIGIT_PREDICTION_UPPER_LIMIT && p < BLANK_PREDICTION_LOWER_LIMIT

const isBlank = p => p >= BLANK_PREDICTION_LOWER_LIMIT
const isDigit = p => p <= DIGIT_PREDICTION_UPPER_LIMIT

const isBlankPredictionCorrect = (label, prediction) =>
  (label && isBlank(prediction)) || (!label && isDigit(prediction))

// convincingly blank => 1
// convincingly digit => 0
// anything else => -1
const normaliseBlankPrediction = prediction =>
  prediction >= BLANK_PREDICTION_LOWER_LIMIT
    ? 1
    : (prediction <= DIGIT_PREDICTION_UPPER_LIMIT ? 0 : -1)

const onPredictBlanks = async () => {
  try {
    SC.hideErrorPanel()
    const parentElement = getPredictionElement('blanks', '.results')
    U.deleteChildren(parentElement)
    const yss = []
    const predictionsArrays = []
    for (const item of testData) {
      const gridImageTensor = await I.loadImage(item.url)
      const { xs, ys, gridSquaresWithDetails } = D.cropGridSquaresFromGrid(item, gridImageTensor)
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
  } catch (error) {
    log.error(`[onPredictBlanks] ${error.message}`)
    SC.showErrorPanel(error.message)
  }
}

const onPredictDigits = async () => {
  try {
    SC.hideErrorPanel()
    const parentElement = getPredictionElement('digits', '.results')
    U.deleteChildren(parentElement)
    const yss = []
    const outputsArray = []
    for (const item of testData) {
      const gridImageTensor = await I.loadImage(item.url)
      const { xs, ys, gridSquaresWithDetails } = D.cropDigitsFromGrid(item, gridImageTensor)
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
  } catch (error) {
    log.error(`[onPredictDigits] ${error.message}`)
    SC.showErrorPanel(error.message)
  }
}

const drawSudokuGrid = (parentElement, indexedDigitPredictions) => {
  const svgElement = DS.createSvgElement('svg', { 'class': 'sudoku-grid' })
  parentElement.appendChild(svgElement)
  const rows = P.indexedDigitPredictionstToRows(indexedDigitPredictions)
  DS.drawInitialGrid(svgElement, rows)
  const initialValues = P.indexedDigitPredictionsToInitialValues(indexedDigitPredictions)
  const solutions = solve(initialValues)
  if (solutions.length === 1) {
    DS.drawSolution(svgElement, solutions[0])
  }
}

const predictBlanksDigitsCommon = async (item, gridImageTensor, boundingBox, parentElement) => {

  const { xs, gridSquaresWithDetails } = D.cropGridSquaresFromKnownGrid(
    gridImageTensor,
    item.puzzleId,
    boundingBox)

  parentElement.appendChild(document.createElement('br'))

  const canvas = await DC.drawGridImageTensor(parentElement, gridImageTensor)
  const ctx = canvas.getContext('2d')

  const blanksPredictionsArray = models.blanks.model.predict(xs).arraySync()

  if (blanksPredictionsArray.some(isBlankPredictionTooInaccurate)) {
    DC.drawBigRedCross(ctx, boundingBox)
    return
  }

  const [blanks, digits] = R.partition(({ index }) =>
    isBlank(blanksPredictionsArray[index]), gridSquaresWithDetails)

  for (const { isBlank, gridSquare } of blanks) {
    const colour = isBlank ? 'green' : 'red'
    DC.drawGridSquare(canvas, gridSquare, colour)
  }

  const xsarr = tf.unstack(xs)
  const indexedDigitPredictions = digits.map(({ index, digit, gridSquare }) => {
    const x = xsarr[index]
    const inputs = tf.stack([x])
    const outputs = models.digits.model.predict(inputs)
    const digitPrediction = outputs.argMax(1).arraySync()[0] + 1
    const colour = digitPrediction === digit ? 'green' : 'red'
    DC.drawGridSquare(canvas, gridSquare, colour)
    return { digitPrediction, index }
  })

  drawSudokuGrid(parentElement, indexedDigitPredictions)
}

// Given bounding boxes, predict blanks then predict digits
// Draw resultant images highlighting correct/incorrect predictions
const onPredictBlanksDigits = async () => {
  try {
    SC.hideErrorPanel()
    const parentElement = getPredictionElement('blanks-digits', '.results')
    U.deleteChildren(parentElement)
    for (const item of testData) {
      const gridImageTensor = await I.loadImage(item.url)
      const boundingBox = item.boundingBox
      await predictBlanksDigitsCommon(item, gridImageTensor, boundingBox, parentElement)
    }
  } catch (error) {
    log.error(`[onPredictBlanksDigits] ${error.message}`)
    SC.showErrorPanel(error.message)
  }
}

// Find bounding boxes then predict blanks then predict digits
// Draw resultant images highlighting correct/incorrect predictions
const onPredictGridBlanksDigits = async () => {
  try {
    SC.hideErrorPanel()
    const parentElement = getPredictionElement('grid-blanks-digits', '.results')
    U.deleteChildren(parentElement)
    for (const item of testData) {
      const gridImageTensor = await I.loadImage(item.url)
      const boundingBox = await findBoundingBox(gridImageTensor)
      await predictBlanksDigitsCommon(item, gridImageTensor, boundingBox, parentElement)
    }
  } catch (error) {
    log.error(`[onPredictGridBlanksDigits] ${error.message}`)
    SC.showErrorPanel(error.message)
  }
}

const onPredictCapture = async () => {
  try {
    performance.clearMarks()
    performance.mark('predict capture start')
    SC.hideErrorPanel()
    const gridImageTensor = I.normaliseGridImage(imageData)
    performance.mark('find bounding box')
    const boundingBox = await findBoundingBox(gridImageTensor)
    const gridSquareImageTensors = D.cropGridSquaresFromUnknownGrid(
      gridImageTensor,
      boundingBox)
    performance.mark('distinguish blanks vs digits')
    const blanksPredictionsArray = models.blanks.model.predict(gridSquareImageTensors).arraySync()
    if (blanksPredictionsArray.some(isBlankPredictionTooInaccurate)) {
      throw new Error('Prediction of blanks vs digits too inaccurate to proceed.')
    }
    const gridSquareImageTensorsArray = tf.unstack(gridSquareImageTensors)
    const indexedDigitImageTensorsArray = gridSquareImageTensorsArray
      .map((digitImageTensor, index) => ({ digitImageTensor, index }))
      .filter(({ index }) => !isBlank(blanksPredictionsArray[index]))
    const digitImageTensorsArray = R.pluck('digitImageTensor', indexedDigitImageTensorsArray)
    const inputs = tf.stack(digitImageTensorsArray)
    performance.mark('recognise digits')
    const outputs = models.digits.model.predict(inputs)
    const digitPredictions = outputs.argMax(1).arraySync().map(R.inc)
    const indexedDigitPredictions = digitPredictions.map((digitPrediction, index) => ({
      digitPrediction,
      index: indexedDigitImageTensorsArray[index].index
    }))
    const parentElement = document.getElementById('predict-capture-results')
    U.deleteChildren(parentElement)
    setWebcamMode(WEBCAM_MODE_SOLUTION)
    drawSudokuGrid(parentElement, indexedDigitPredictions)
    performance.mark('predict capture end')
    const marks = performance.getEntriesByType('mark')
    marks.forEach(mark => log.info(JSON.stringify(mark)))
    const transformedMarks = marks
      .map(mark => R.pick(['name', 'startTime'], mark))
      .map(({ name, startTime }) => ({ name, startTime: startTime - marks[0].startTime }))
    const messageArea = document.getElementById('message-area')
    messageArea.innerText = JSON.stringify(transformedMarks, null, 2)
    performance.clearMarks()
  } catch (error) {
    log.error(`[onPredictCapture] ${error.message}`)
    SC.showErrorPanel(error.message)
  }
}

const onSaveModel = name => async () => {
  try {
    SC.hideErrorPanel()
    log.info(`Saving model ${name}...`)
    const saveResult = await models[name].model.save(`${location.origin}/api/saveModel/${name}`)
    log.info(`saveResult: ${JSON.stringify(saveResult)}`)
  } catch (error) {
    log.error(`[onSaveModel(${name})] ${error.message}`)
    SC.showErrorPanel(error.message)
  }
}

const onLoadModel = name => async () => {
  try {
    SC.hideErrorPanel()
    log.info(`Loading model ${name}...`)
    models[name].model = await tf.loadLayersModel(`${location.origin}/models/${name}/model.json`)
    models[name].trained = true
  } catch (error) {
    log.error(`[onLoadModel(${name})] ${error.message}`)
    SC.showErrorPanel(error.message)
  }
}

const updateButtonStates = () => {

  const saveBlanksBtn = getTrainingElement('blanks', '.save-btn')
  const saveDigitsBtn = getTrainingElement('digits', '.save-btn')

  saveBlanksBtn.disabled = !models.blanks.trained
  saveDigitsBtn.disabled = !models.digits.trained

  const predictBlanksBtn = getPredictionElement('blanks', '.predict-btn')
  const predictDigitsBtn = getPredictionElement('digits', '.predict-btn')
  const predictBlanksDigitsBtn = getPredictionElement('blanks-digits', '.predict-btn')
  const predictGridBlanksDigitsBtn = getPredictionElement('grid-blanks-digits', '.predict-btn')

  predictBlanksBtn.disabled = !models.blanks.trained
  predictDigitsBtn.disabled = !models.digits.trained
  predictBlanksDigitsBtn.disabled = !(models.blanks.trained && models.digits.trained)
  predictGridBlanksDigitsBtn.disabled = !(models.blanks.trained && models.digits.trained)
  predictCaptureBtn.disabled = !(models.blanks.trained && models.digits.trained && imageData)

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

const onIdle = () => {
  updateButtonStates()
  requestAnimationFrame(onIdle)
}

SC.addTrainingSection('blanks', onTrainBlanks, onSaveModel, onLoadModel)
SC.addTrainingSection('digits', onTrainDigits, onSaveModel, onLoadModel)

SC.addPredictionSection('grid', onPredictGrid)
SC.addPredictionSection('blanks', onPredictBlanks)
SC.addPredictionSection('digits', onPredictDigits)
SC.addPredictionSection('blanks-digits', onPredictBlanksDigits)
SC.addPredictionSection('grid-blanks-digits', onPredictGridBlanksDigits)

document.querySelectorAll('.clear-btn').forEach(clearBtn =>
  clearBtn.addEventListener('click', () => visor && visor.close()))

const predictCaptureBtn = document.getElementById('predict-capture-btn')
predictCaptureBtn.addEventListener('click', onPredictCapture)

const showVisorBtn = document.getElementById('show-visor-btn')
showVisorBtn.addEventListener('click', () => visor && visor.open())

const main = async () => {
  drawGuides()
  initialiseCamera()
  onIdle()
}

main()
