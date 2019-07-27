import * as tf from '@tensorflow/tfjs'
import * as R from 'ramda'
import axios from 'axios'
import trainingData from '../data/training-data.json'

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

const drawImageTensors = (trainData, imageTensors) => {
  const body = document.querySelector('body')
  imageTensors.forEach(async (imageTensor, index) => {
    const { boundingBox } = trainData[index]
    const canvas = document.createElement('canvas')
    await tf.browser.toPixels(imageTensor, canvas)
    const ctx = canvas.getContext('2d')
    ctx.strokeStyle = 'blue'
    ctx.strokeRect(...boundingBox)
    drawDigitBoxes(ctx, boundingBox)
    body.appendChild(canvas)
  })
}

const loadImage = url =>
  new Promise(resolve => {
    const image = new Image()
    image.onload = () => resolve(tf.browser.fromPixels(image, 1))
    image.src = url
  })

const loadData = async data => {
  const urls = data.map(datum => datum.url)
  const promises = urls.map(loadImage)
  const imageTensors = await Promise.all(promises)
  drawImageTensors(data, imageTensors)
  const xs = tf.stack(imageTensors)
  const ys = tf.tensor2d(data.map(datum => datum.boundingBox), undefined, 'int32')
  console.log(`xs - rank: ${xs.rank}; shape: ${xs.shape}; dtype: ${xs.dtype}`)
  console.log(`ys - rank: ${ys.rank}; shape: ${ys.shape}; dtype: ${ys.dtype}`)
  return { xs, ys }
}

const loadTrainingData = () => loadData(trainingData)
// const loadTestData = () => loadData(testData)

const createModel = trainingData => {
  const [, W, H, C] = trainingData.xs.shape
  const inputShape = [W, H, C]
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

const train = async (model, trainingData) => {
  model.compile({
    optimizer: 'rmsprop',
    loss: 'meanSquaredError'
  })
  const params = {
    // batchSize: 10,
    // validationSplit: 0.2,
    batchSize: 1,
    validationSplit: 0,
    epochs: 10
  }
  // TODO: later, use model.fitDataset() ?
  return model.fit(trainingData.xs, trainingData.ys, params)
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

  const updateButtonState = playing => {
    startBtn.disabled = playing
    stopBtn.disabled = !playing
    captureBtn.disabled = !playing
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
      updateButtonState(true)
    }
  }

  const onStop = () => {
    const mediaStream = videoElement.srcObject
    mediaStream.getVideoTracks()[0].stop()
    videoElement.srcObject = null
    updateButtonState(false)
  }

  const onCapture = async () => {
    const imageBitmap = await createImageBitmap(videoElement)
    capturedImageElementContext.drawImage(imageBitmap, 0, 0)
    onStop()
  }

  const onSave = async () => {
    const dataUrl = capturedImageElement.toDataURL('image/png')
    const response = await axios.post('/api/saveImage', { dataUrl })
    messageArea.innerText = response.data
  }

  const onClear = () => {
    capturedImageElementContext.clearRect(0, 0, capturedImageElement.width, capturedImageElement.height)
    messageArea.innerText = ''
  }

  startBtn.addEventListener('click', onStart)
  stopBtn.addEventListener('click', onStop)
  captureBtn.addEventListener('click', onCapture)
  saveBtn.addEventListener('click', onSave)
  clearBtn.addEventListener('click', onClear)

  updateButtonState(false)
}

const onTrain = async () => {
  try {
    trainBtn.disabled = true
    const trainData = await loadTrainingData()
    const model = createModel(trainData)
    const history = await train(model, trainData)
    console.dir(history)
    // const testData = await getTestData()
    // const testResult = model.evaluate(testData.xs, testData.labels)
  } finally {
    trainBtn.disabled = false
  }
}

const trainBtn = document.getElementById('trainBtn')
trainBtn.addEventListener('click', onTrain)

const main = async () => {
  drawGuides()
  initialiseCamera()
}

main()
