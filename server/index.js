const express = require('express')
const bodyParser = require('body-parser')
const path = require('path')
const fs = require('fs').promises
const PORT = process.env.PORT || 3060
const distFolder = path.resolve(__dirname, '..', 'dist')
const rawImagesFolder = path.resolve(__dirname, '..', 'scanned-images', 'raw')
const app = express()
const apiRouter = express.Router()
const PNG_EXT = '.png'
const getNextFileName = async () => {
  const fileNames = await fs.readdir(rawImagesFolder)
  const numbers = fileNames
    .filter(fileName => fileName.endsWith(PNG_EXT))
    .map(fileName => path.basename(fileName, PNG_EXT))
    .map(Number)
    .filter(Number.isInteger)
  const biggestNumber = numbers.length ? Math.max(...numbers) : 0
  const nextNumber = biggestNumber + 1
  const nextNumberPadded = nextNumber.toString().padStart(5, 0)
  return path.resolve(rawImagesFolder, `${nextNumberPadded}${PNG_EXT}`)
}
const onSaveImage = async (req, res) => {
  try {
    const dataUrl = req.body.dataUrl
    console.log(`[onSaveImage] dataUrl.length: ${dataUrl.length}`)
    const data = dataUrl.replace(/^data:image\/png;base64,/, '')
    const buffer = Buffer.from(data, 'base64')
    const fileName = await getNextFileName()
    console.log(`[onSaveImage] fileName: ${fileName}`)
    await fs.writeFile(fileName, buffer)
    res.status(201).send(`Saved image to ${fileName}.`)
  } catch (error) {
    console.log(`[onSaveImage] ERROR: ${error.message}`)
  }
}
apiRouter.post('/saveImage', onSaveImage)
app.use(express.static(distFolder))
app.use(bodyParser.json({ limit: '1MB' }))
app.use('/api', apiRouter)
app.listen(PORT, () => console.log(`Listening on http://localhost:${PORT}`))
