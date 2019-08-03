const express = require('express')
const path = require('path')
const fs = require('fs').promises

const PNG_EXT = '.png'

const configureApiRouter = (rawImagesFolder, normalisedImagesFolder) => {

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
      console.log(`[saveImage] buffer.length: ${buffer.length}`)
      const fileName = await getNextFileName(folder)
      console.log(`[saveImage] saving image to ${fileName}`)
      await fs.writeFile(fileName, buffer)
      res.status(201).send(`Saved image to ${fileName}.`)
    } catch (error) {
      console.log(`[saveImage] ERROR: ${error.message}`)
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

  const onListRawImages = async (_, res) =>
    listImages(res, rawImagesFolder)

  const onListNormalisedImages = async (_, res) =>
    listImages(res, normalisedImagesFolder)

  const apiRouter = express.Router()
  apiRouter.post('/saveRawImage', onSaveRawImage)
  apiRouter.post('/saveNormalisedImage', onSaveNormalisedImage)
  apiRouter.get('/listRawImages', onListRawImages)
  apiRouter.get('/listNormalisedImages', onListNormalisedImages)
  return apiRouter
}

module.exports = {
  configureApiRouter
}
