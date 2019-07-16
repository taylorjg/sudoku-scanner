import * as tf from '@tensorflow/tfjs'
import axios from 'axios'

const IMAGE_WIDTH = 224
const IMAGE_HEIGHT = 224
const CHANNELS = 1

const getTrainData = async () => {
  // Tensor4D: [numTrainExamples, 224, 224, 1]
  const xs = []
  // Tensor2D: [numTrainExamples, 4]
  const labels = []
  return {
    xs,
    labels
  }
}

const createModel = () => {
  const inputShape = [IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS]
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
  const trainData = await getTrainData()
  model.compile({
    optimizer: 'rmsprop',
    loss: 'meanSquaredError'
  })
  const args = {
    batchSize: 10,
    validationSplit: 0.2,
    epochs: 10
  }
  // TODO: later, use model.fitDataset() ?
  /* const history = */ await model.fit(trainData.xs, trainData.labels, args)
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
  const w = svg.getBoundingClientRect().width
  const h = svg.getBoundingClientRect().height
  const ix = w * 0.05 // inset x
  const iy = h * 0.05 // inset y
  const ax = w * 0.1 // arm x
  const ay = h * 0.1 // arm y
  svg.appendChild(createVideoGuide(`M${ix + ax},${iy} h${-ax} v${ay}`))
  svg.appendChild(createVideoGuide(`M${w - ix - ax},${iy} h${ax} v${ay}`))
  svg.appendChild(createVideoGuide(`M${ix + ax},${h - iy} h${-ax} v${-ay}`))
  svg.appendChild(createVideoGuide(`M${w - ix - ax},${h - iy} h${ax} v${-ay}`))
}

const initialiseVideoCapture = async () => {

  const videoElement = document.getElementById('video')
  const videoElementRect = videoElement.getBoundingClientRect()
  const capturedImageElement = document.getElementById('captured-image')
  capturedImageElement.width = videoElementRect.width
  capturedImageElement.height = videoElementRect.height
  const capturedImageElementContext = capturedImageElement.getContext('2d')
  const startBtn = document.getElementById('startBtn')
  const stopBtn = document.getElementById('stopBtn')
  const captureBtn = document.getElementById('captureBtn')
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
      console.log(mediaStream)
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
    const dataUrl = capturedImageElement.toDataURL('image/png')
    const response = await axios.post('/api/saveImage', { dataUrl })
    messageArea.innerText = response.data
    onStop()
  }

  const onClear = () => {
    capturedImageElementContext.clearRect(0, 0, capturedImageElement.width, capturedImageElement.height)
  }

  startBtn.addEventListener('click', onStart)
  stopBtn.addEventListener('click', onStop)
  captureBtn.addEventListener('click', onCapture)
  clearBtn.addEventListener('click', onClear)

  updateButtonState(false)
}

const main = async () => {
  drawGuides()
  await initialiseVideoCapture()
  // const model = createModel()
  // await train(model)
  // const testData = await getTestData()
  // const testResult = model.evaluate(testData.xs, testData.labels)
}

main()
