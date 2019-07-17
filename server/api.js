const express = require('express')
const path = require('path')
const fs = require('fs').promises

const PNG_EXT = '.png'

const configureApiRouter = rawImagesFolder => {

  const numberToFileName = n =>
    `${n.toString().padStart(5, 0)}${PNG_EXT}`

  const getNextFileName = async () => {
    const fileNames = await fs.readdir(rawImagesFolder)
    const numbers = fileNames
      .filter(fileName => fileName.endsWith(PNG_EXT))
      .map(fileName => path.basename(fileName, PNG_EXT))
      .map(Number)
      .filter(Number.isInteger)
    const biggestNumber = numbers.length ? Math.max(...numbers) : 0
    const nextNumber = biggestNumber + 1
    return path.resolve(rawImagesFolder, numberToFileName(nextNumber))
  }

  const onSaveImage = async (req, res) => {
    try {
      const dataUrl = req.body.dataUrl
      const data = dataUrl.replace(/^data:image[/]png;base64,/, '')
      const buffer = Buffer.from(data, 'base64')
      console.log(`[onSaveImage] buffer.length: ${buffer.length}`)
      const fileName = await getNextFileName()
      console.log(`[onSaveImage] saving image to ${fileName}`)
      await fs.writeFile(fileName, buffer)
      res.status(201).send(`Saved image to ${fileName}.`)
    } catch (error) {
      console.log(`[onSaveImage] ERROR: ${error.message}`)
    }
  }

  const onListImages = async (req, res) => {
    const fileNames = await fs.readdir(rawImagesFolder)
    const filteredFileNames = fileNames
      .filter(fileName => fileName.endsWith(PNG_EXT))
      .filter(fileName => {
        const basename = path.basename(fileName, PNG_EXT)
        const number = Number(basename)
        return Number.isInteger(number)
      })
    res.json(filteredFileNames)
  }

  const apiRouter = express.Router()
  apiRouter.post('/saveImage', onSaveImage)
  apiRouter.get('/listImages', onListImages)
  return apiRouter
}

module.exports = {
  configureApiRouter
}
