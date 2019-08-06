import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import * as R from 'ramda'
import axios from 'axios'
import puzzles from '../data/puzzles.json'
import trainingData from '../data/training-data.json'
import validationData from '../data/validation-data.json'
import testData from '../data/test-data.json'

let gridModel = undefined
let digitsModel = undefined
let trainedGrid = false
let trainedDigits = false
let imageData = undefined

const IMAGE_WIDTH = 224
const IMAGE_HEIGHT = 224
const IMAGE_CHANNELS = 1

const inset = (x, y, w, h, dx, dy) =>
  [x + dx, y + dy, w - 2 * dx, h - 2 * dy]

function* calculateDigitBoxes(boundingBox) {
  const [bbx, bby, bbw, bbh] = boundingBox
  const w = bbw / 9
  const h = bbh / 9
  const dx = w / 10
  const dy = h / 10
  for (const row of R.range(0, 9)) {
    const y = bby + row * h
    for (const col of R.range(0, 9)) {
      const x = bbx + col * w
      yield inset(x, y, w, h, dx, dy)
    }
  }
}

const drawDigitBoxes = (ctx, boundingBox, colour) => {
  for (const digitBox of calculateDigitBoxes(boundingBox)) {
    ctx.strokeStyle = colour
    ctx.strokeRect(...digitBox)
  }
}

const drawImageTensor = async (imageTensor, boundingBoxTarget, boundingBoxPrediction) => {
  const canvas = document.createElement('canvas')
  await tf.browser.toPixels(imageTensor, canvas)
  const ctx = canvas.getContext('2d')
  if (boundingBoxTarget) {
    ctx.strokeStyle = 'blue'
    ctx.strokeRect(...boundingBoxTarget)
  }
  if (boundingBoxPrediction) {
    ctx.strokeStyle = 'red'
    ctx.strokeRect(...boundingBoxPrediction)
    drawDigitBoxes(ctx, boundingBoxPrediction, 'red')
  }
  const body = document.querySelector('body')
  body.appendChild(canvas)
}

const drawDigitImageTensors = t4d => {
  const body = document.querySelector('body')
  const t3ds = tf.unstack(t4d)
  t3ds.forEach(async t3d => {
    const canvas = document.createElement('canvas')
    canvas.setAttribute('style', 'margin: 2px')
    await tf.browser.toPixels(t3d, canvas)
    body.appendChild(canvas)
  })
}

// const drawImageTensors = (trainData, imageTensors) => {
//   const body = document.querySelector('body')
//   imageTensors.forEach(async (imageTensor, index) => {
//     drawImageTensor(imageTensor)
//     const { boundingBox } = trainData[index]
//     const canvas = document.createElement('canvas')
//     await tf.browser.toPixels(imageTensor, canvas)
//     const ctx = canvas.getContext('2d')
//     ctx.strokeStyle = 'blue'
//     ctx.strokeRect(...boundingBox)
//     drawDigitBoxes(ctx, boundingBox)
//     body.appendChild(canvas)
//   })
// }

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

const IMAGE_CACHE = new Map()

const loadImage = async url => {
  const existingImageTensor = IMAGE_CACHE.get(url)
  if (existingImageTensor) return existingImageTensor
  const promise = new Promise(resolve => {
    console.log(`Loading ${url}`)
    const image = new Image()
    image.onload = () => resolve(tf.browser.fromPixels(image, IMAGE_CHANNELS))
    image.src = url
  })
  const imageTensor = await promise
  IMAGE_CACHE.set(url, imageTensor)
  return imageTensor
}

// const loadImage2 = async url => {
//   const existingImageTensor = IMAGE_CACHE.get(url)
//   if (existingImageTensor) return existingImageTensor
//   const promise = new Promise(resolve => {
//     console.log(`Loading ${url}`)
//     const image = new Image()
//     image.onload = () => {
//       const canvas = document.createElement('canvas')
//       canvas.width = image.width
//       canvas.height = image.height
//       const ctx = canvas.getContext('2d')
//       ctx.drawImage(image, 0, 0)
//       const imageData = ctx.getImageData(0, 0, image.width, image.height)
//       const imageDataGreyscale = convertToGreyscale(imageData)
//       return resolve(tf.browser.fromPixels(imageDataGreyscale, IMAGE_CHANNELS))
//     }
//     image.src = url
//   })
//   const imageTensor = await promise
//   IMAGE_CACHE.set(url, imageTensor)
//   return imageTensor
// }

// const loadData = async data => {
//   const urls = data.map(datum => datum.url)
//   const promises = urls.map(loadImage)
//   const imageTensors = await Promise.all(promises)
//   drawImageTensors(data, imageTensors)
//   const xs = tf.stack(imageTensors)
//   const ys = tf.tensor2d(data.map(datum => datum.boundingBox), undefined, 'int32')
//   console.log(`xs - rank: ${xs.rank}; shape: ${xs.shape}; dtype: ${xs.dtype}`)
//   console.log(`ys - rank: ${ys.rank}; shape: ${ys.shape}; dtype: ${ys.dtype}`)
//   return { xs, ys }
// }

// const loadTrainingData = () => loadData(trainingData)

const GRID_DATA_BATCH_SIZE = 1000

async function* gridDataGenerator(data) {
  const batches = R.splitEvery(GRID_DATA_BATCH_SIZE, data)
  for (const batch of batches) {
    const urls = batch.map(item => item.url)
    const promises = urls.map(loadImage)
    const imageTensors = await Promise.all(promises)
    // imageTensors.forEach((imageTensor, index) => {
    //   const boundingBox = batch[index].boundingBox
    //   drawImageTensor(imageTensor, boundingBox)
    // })
    const xs = tf.stack(imageTensors)
    const ys = tf.tensor2d(batch.map(item => item.boundingBox), undefined, 'int32')
    yield { xs, ys }
  }
}

const digitDataGenerator = async data => {
  const urls = data.map(item => item.url)
  const promises = urls.map(loadImage)
  const imageTensors = await Promise.all(promises)
  imageTensors.forEach((imageTensor, index) => {
    const { boundingBox, puzzleId } = data[index]
    const puzzle = puzzles.find(p => p.id === puzzleId)
    const initialValues = Array.from(puzzle.initialValues.join(''))
    const digits = initialValues
      .map((initialValue, index) => ({ digit: Number(initialValue), index }))
      .filter(({ digit }) => Number.isInteger(digit) && digit >= 1 && digit <= 9)
    const allDigitBoxes = Array.from(calculateDigitBoxes(boundingBox))
    const digitBoxes = digits
      .map(({ digit, index }) => ({ digit, digitBox: allDigitBoxes[index] }))
    const image = tf.stack([imageTensor.toFloat().div(255)])
    const boxes = digitBoxes.map(({ digitBox }) => {
      const [x, y, w, h] = digitBox
      const y1 = y
      const x1 = x
      const y2 = y1 + h
      const x2 = x1 + w
      return [
        y1 / (IMAGE_HEIGHT - 1),
        x1 / (IMAGE_WIDTH - 1),
        y2 / (IMAGE_HEIGHT - 1),
        x2 / (IMAGE_WIDTH - 1)
      ]
    })
    const boxInd = Array(boxes.length).fill(0)
    const cropSize = [20, 20]
    const digitImageTensors = tf.image.cropAndResize(image, boxes, boxInd, cropSize)
    drawDigitImageTensors(digitImageTensors)
  })
}

const createGridModel = () => {
  const inputShape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
  const conv2dArgs = {
    kernelSize: 3,
    filters: 32,
    activation: 'sigmoid'
  }
  const maxPooling2dArgs = {
    poolSize: 2,
    strides: 2
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

  model.add(tf.layers.dense({ units: 128, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 4 }))

  model.summary()

  return model
}

const createDigitsModel = () => {
}

const trainGrid = async model => {
  model.compile({
    optimizer: 'rmsprop',
    loss: 'meanSquaredError'
    // loss: 'meanAbsoluteError'
  })
  const trainingDataset = tf.data.generator(() => gridDataGenerator(trainingData))
  const validationDataset = tf.data.generator(() => gridDataGenerator(validationData))

  const trainingSurface = tfvis.visor().surface({ tab: 'Tab 1', name: 'Model Training' })
  const customCallback = tfvis.show.fitCallbacks(
    trainingSurface,
    ['loss', 'val_loss'],
    {
      callbacks: ['onBatchEnd', 'onEpochEnd']
    })

  const params = {
    epochs: 20,
    validationData: validationDataset,
    callbacks: customCallback
  }
  return model.fitDataset(trainingDataset, params)
}

const trainDigits = async model => {
}

const createSvgElement = (elementName, additionalAttributes = {}) => {
  const element = document.createElementNS('http://www.w3.org/2000/svg', elementName)
  for (const [name, value] of Object.entries(additionalAttributes)) {
    element.setAttribute(name, value)
  }
  return element
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
    predictCaptureBtn.disabled = !imageData // || !trained
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
    canvas.width = IMAGE_WIDTH
    canvas.height = IMAGE_HEIGHT
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
    predictGridTestDataBtn.disabled = false
  } finally {
    trainGridBtn.disabled = false
  }
}

const onTrainDigits = async () => {
  try {
    trainDigitsBtn.disabled = true
    await digitDataGenerator(trainingData)
    // digitsModel = createDigitsModel()
    // const trainingResults = await trainDigits(digitsModel)
    // console.dir(trainingResults)
    // const lastLoss = R.last(trainingResults.history.loss)
    // const lastValLoss = R.last(trainingResults.history.val_loss)
    // console.log('last loss:', lastLoss, 'sqrt:', Math.sqrt(lastLoss))
    // console.log('last val_loss:', lastValLoss, 'sqrt:', Math.sqrt(lastValLoss))
    // trainedDigits = true
    // predictDigitsTestDataBtn.disabled = false
  } finally {
    trainDigitsBtn.disabled = false
  }
}

const normaliseGridImage = imageData => {
  const imageDataGreyscale = convertToGreyscale(imageData)
  const imageTensorGreyscale = tf.browser.fromPixels(imageDataGreyscale, IMAGE_CHANNELS)
  return tf.image.resizeBilinear(imageTensorGreyscale, [IMAGE_HEIGHT, IMAGE_WIDTH])
}

const onPredictCapture = async () => {
  const imageTensor = normaliseGridImage(imageData)
  // const input = tf.stack([imageTensor])
  // const output = model.predict(input)
  // const boundingBoxPrediction = output.arraySync()[0]
  // console.log(`boundingBoxPrediction: ${JSON.stringify(boundingBoxPrediction)}`)
  // drawImageTensor(imageTensor, undefined, boundingBoxPrediction)
  drawImageTensor(imageTensor, undefined, undefined)
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
    drawImageTensor(imageTensor, boundingBoxTarget, boundingBoxPrediction)
  })
}

const onPredictDigitsTestData = async () => {
}

const trainGridBtn = document.getElementById('trainGridBtn')
trainGridBtn.addEventListener('click', onTrainGrid)

const trainDigitsBtn = document.getElementById('trainDigitsBtn')
trainDigitsBtn.addEventListener('click', onTrainDigits)

const predictCaptureBtn = document.getElementById('predictCaptureBtn')
predictCaptureBtn.addEventListener('click', onPredictCapture)
predictCaptureBtn.disabled = true

const predictGridTestDataBtn = document.getElementById('predictGridTestDataBtn')
predictGridTestDataBtn.addEventListener('click', onPredictGridTestData)
predictGridTestDataBtn.disabled = true

const predictDigitsTestDataBtn = document.getElementById('predictDigitsTestDataBtn')
predictDigitsTestDataBtn.addEventListener('click', onPredictDigitsTestData)
predictDigitsTestDataBtn.disabled = true

const main = async () => {
  drawGuides()
  initialiseCamera()
}

main()
