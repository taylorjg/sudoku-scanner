const express = require('express')
const path = require('path')
const fs = require('fs').promises
const multer = require('multer')
const log = require('loglevel')

const PNG_EXT = '.png'

const configureApiRouter = (rawImagesFolder, normalisedImagesFolder, modelsFolder) => {

  const myDiskStorage = multer.diskStorage({
    destination: function (req, _file, cb) {
      const model = req.params.model
      const dest = model
        ? path.resolve(modelsFolder, model)
        : modelsFolder
      log.info(`[multer.diskStorage#destination] model: ${model}; dest: ${dest}`)
      cb(null, dest)
    },
    filename: function (_req, file, cb) {
      cb(null, file.fieldname)
    }
  })

  const upload = multer({ storage: myDiskStorage })

  const numberToFileName = n =>
    `${n.toString().padStart(5, 0)}${PNG_EXT}`

  const getNextFileName = async folder => {
    const fileNames = await fs.readdir(folder)
    const numbers = fileNames
      .filter(fileName => fileName.endsWith(PNG_EXT))
      .map(fileName => path.basename(fileName, PNG_EXT))
      .map(Number)
      .filter(Number.isInteger)
    const biggestNumber = numbers.length ? Math.max(...numbers) : 0
    const nextNumber = biggestNumber + 1
    return path.resolve(folder, numberToFileName(nextNumber))
  }

  const saveImage = async (req, res, folder) => {
    try {
      const dataUrl = req.body.dataUrl
      const data = dataUrl.replace(/^data:image[/]png;base64,/, '')
      const buffer = Buffer.from(data, 'base64')
      log.info(`[saveImage] buffer.length: ${buffer.length}`)
      const fileName = await getNextFileName(folder)
      log.info(`[saveImage] saving image to ${fileName}`)
      await fs.writeFile(fileName, buffer)
      res.status(201).send(fileName)
    } catch (error) {
      log.error(`[saveImage] ${error.message}`)
    }
  }

  const onSaveRawImage = async (req, res) =>
    saveImage(req, res, rawImagesFolder)

  const onSaveNormalisedImage = async (req, res) =>
    saveImage(req, res, normalisedImagesFolder)

  const listImages = async (res, folder) => {
    const fileNames = await fs.readdir(folder)
    const filteredFileNames = fileNames
      .filter(fileName => fileName.endsWith(PNG_EXT))
      .map(fileName => path.resolve(folder, fileName))
    res.json(filteredFileNames)
  }

  const onListRawImages = async (_req, res) =>
    listImages(res, rawImagesFolder)

  const onListNormalisedImages = async (_req, res) =>
    listImages(res, normalisedImagesFolder)

  const onSaveModel = async (_req, res) => {
    res.end()
  }

  const fields = [
    { name: 'model.json', maxCount: 1 },
    { name: 'model.weights.bin', maxCount: 1 }
  ]

  const apiRouter = express.Router()
  apiRouter.post('/saveRawImage', onSaveRawImage)
  apiRouter.post('/saveNormalisedImage', onSaveNormalisedImage)
  apiRouter.get('/listRawImages', onListRawImages)
  apiRouter.get('/listNormalisedImages', onListNormalisedImages)
  apiRouter.post('/saveModel/:model', upload.fields(fields), onSaveModel)
  return apiRouter
}

module.exports = {
  configureApiRouter
}
