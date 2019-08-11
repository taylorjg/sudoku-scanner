const path = require('path')
const express = require('express')
const bodyParser = require('body-parser')
const { configureApiRouter } = require('./api')

const PORT = process.env.PORT || 3060
const DIST_FOLDER = path.resolve(__dirname, '..', 'dist')
const RAW_IMAGES_FOLDER = path.resolve(__dirname, '..', 'scanned-images', 'raw')
const NORMALISED_IMAGES_FOLDER = path.resolve(__dirname, '..', 'scanned-images', 'normalised')
const MODELS_FOLDER = path.resolve(__dirname, '..', 'models')

const apiRouter = configureApiRouter(RAW_IMAGES_FOLDER, NORMALISED_IMAGES_FOLDER, MODELS_FOLDER)

const app = express()
app.use(express.static(DIST_FOLDER))
app.use('/rawImages', express.static(RAW_IMAGES_FOLDER))
app.use('/normalisedImages', express.static(NORMALISED_IMAGES_FOLDER))
app.use('/models', express.static(MODELS_FOLDER))
app.use(bodyParser.json({ limit: '1MB' }))
app.use('/api', apiRouter)
app.listen(PORT, () => console.log(`Listening on http://localhost:${PORT}`))
