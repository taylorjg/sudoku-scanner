const express = require('express')
const bodyParser = require('body-parser')
const path = require('path')
const fs = require('fs').promises
const PORT = process.env.PORT || 3060
const distFolder = path.resolve(__dirname, '..', 'dist')
const app = express()
const apiRouter = express.Router()
const onSaveImage = async (req, res) => {
  try {
    const dataUrl = req.body.dataUrl
    console.log(`[onSaveImage] dataUrl.length: ${dataUrl.length}`)
    const data = dataUrl.replace(/^data:image\/png;base64,/, '')
    const buffer = Buffer.from(data, 'base64')
    await fs.writeFile('image.png', buffer)
    res.end()
  } catch (error) {
    console.log(`[onSaveImage] ERROR: ${error.message}`)
  }
}
apiRouter.post('/saveImage', onSaveImage)
app.use(express.static(distFolder))
app.use(bodyParser.json({ limit: '1MB' }))
app.use('/api', apiRouter)
app.listen(PORT, () => console.log(`Listening on http://localhost:${PORT}`))
