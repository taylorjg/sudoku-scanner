import * as tf from '@tensorflow/tfjs'
import * as R from 'ramda'
import axios from 'axios'
import trainingData from '../data/training-data.json'

let trained = false
let model = undefined
let imageBitmap = undefined
let imageData = undefined

const IMAGE_WIDTH = 224
const IMAGE_HEIGHT = 224
const IMAGE_CHANNELS = 1

const inset = (x, y, w, h, dx, dy) =>
  [x + dx, y + dy, w - 2 * dx, h - 2 * dy]

const drawDigitBoxes = (ctx, boundingBox) => {
  const [bbx, bby, bbw, bbh] = boundingBox
  const digitw = bbw / 9
  const digith = bbh / 9
  for (const col of R.range(0, 9)) {
    const x = bbx + col * digitw
    for (const row of R.range(0, 9)) {
      const y = bby + row * digith
      ctx.strokeStyle = 'red'
      ctx.strokeRect(...inset(x, y, digitw, digith, digitw / 10, digith / 10))
    }
  }
}

const drawImageTensor = async (imageTensor, boundingBox) => {
  const canvas = document.createElement('canvas')
  await tf.browser.toPixels(imageTensor, canvas)
  const ctx = canvas.getContext('2d')
  ctx.strokeStyle = 'blue'
  ctx.strokeRect(...boundingBox)
  drawDigitBoxes(ctx, boundingBox)
  const body = document.querySelector('body')
  body.appendChild(canvas)
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

const BATCH_SIZE = 3

async function* trainingDataGenerator() {
  tf.util.shuffle(trainingData)
  const batches = R.splitEvery(BATCH_SIZE, trainingData)
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

const createModel = () => {
  const inputShape = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]
  const model = tf.sequential()
  model.add(tf.layers.conv2d({ inputShape, kernelSize: 3, filters: 16, activation: 'relu' }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }))
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 4 }))
  model.summary()
  return model
}

const train = async model => {
  model.compile({
    optimizer: 'rmsprop',
    loss: 'meanSquaredError'
  })
  const params = {
    epochs: 10
  }
  const ds = tf.data.generator(trainingDataGenerator)
  return model.fitDataset(ds, params)
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
    saveBtn.disabled = !imageBitmap
    clearBtn.disabled = !imageBitmap
    // predictBtn.disabled = !imageBitmap || !trained
    predictBtn.disabled = !imageData
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
    imageBitmap = await createImageBitmap(videoElement)
    capturedImageElementContext.drawImage(imageBitmap, 0, 0)
    const w = capturedImageElementContext.canvas.width
    const h = capturedImageElementContext.canvas.height
    imageData = capturedImageElementContext.getImageData(0, 0, w, h)
    onStop()
  }

  const onSave = async () => {
    const dataUrl = capturedImageElement.toDataURL('image/png')
    const response = await axios.post('/api/saveImage', { dataUrl })
    messageArea.innerText = response.data
  }

  const onClear = () => {
    capturedImageElementContext.clearRect(0, 0, capturedImageElement.width, capturedImageElement.height)
    imageBitmap = undefined
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

const onTrain = async () => {
  try {
    trainBtn.disabled = true
    model = createModel()
    const history = await train(model)
    console.dir(history)
    trained = true
    // const testData = await getTestData()
    // const testResult = model.evaluate(testData.xs, testData.labels)
  } finally {
    trainBtn.disabled = false
  }
}

const convertToGreyscale = imageData => {
  const w = imageData.width
  const h = imageData.height
  const numPixels = w * h
  const data = imageData.data
  const array = new Uint8ClampedArray(data.length)
  const steps = R.range(0, numPixels).map(index => index * 4)
  for (const step of steps) {
    const r = data[step]
    const g = data[step + 1]
    const b = data[step + 2]
    const avg = (r + b + g) / 3
    array[step] = avg
    array[step + 1] = avg
    array[step + 2] = avg
    array[step + 3] = 255
  }
  return new ImageData(array, w, h)
}

const onPredict = async () => {
  const imageDataGreyscale = convertToGreyscale(imageData)
  const imageTensor = tf.browser.fromPixels(imageDataGreyscale, IMAGE_CHANNELS)
  const imageTensorResized = tf.image.resizeBilinear(imageTensor, [IMAGE_WIDTH, IMAGE_HEIGHT])
  const input = tf.stack([imageTensorResized])
  const output = model.predict(input)
  const boundingBox = output.arraySync()[0]
  console.log(`boundingBox: ${JSON.stringify(boundingBox)}`)
  drawImageTensor(imageTensorResized, boundingBox)
}

const trainBtn = document.getElementById('trainBtn')
trainBtn.addEventListener('click', onTrain)

const predictBtn = document.getElementById('predictBtn')
predictBtn.addEventListener('click', onPredict)
// predictBtn.disabled = true

const main = async () => {
  drawGuides()
  initialiseCamera()
}

main()
