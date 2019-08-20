const fs = require('fs').promises
const path = require('path')
const R = require('ramda')
const axios = require('axios')
const log = require('loglevel')

const RAW_IMAGES_FOLDER = path.resolve(__dirname, '..', 'scanned-images', 'raw')
const NORMALISED_IMAGES_FOLDER = path.resolve(__dirname, '..', 'scanned-images', 'normalised')
const PNG_EXT = '.png'

const stringToIntegerOrUndefined = s => {
  const n = Number(s)
  return Number.isInteger(n) ? n : undefined
}

const numberToFileName = n =>
  `${n.toString().padStart(5, 0)}${PNG_EXT}`

const axiosInstance = axios.create({
  baseURL: 'https://sudoku-scanner.herokuapp.com',
  responseType: 'arraybuffer'
})

const downloadImages = async (fileNames, sourcePath, destinationFolder) => {
  const urlPromises = fileNames.map(fileName => {
    const url = `${sourcePath}/${fileName}`
    log.info(`Fetching ${url}`)
    return axiosInstance.get(url)
  })
  const responses = await Promise.all(urlPromises)
  const filePromises = responses.map((response, index) => {
    const fileName = fileNames[index]
    const fullFileName = path.resolve(destinationFolder, fileName)
    log.info(`Writing ${fullFileName}`)
    return fs.writeFile(fullFileName, response.data)
  })
  await Promise.all(filePromises)
}

const main = async () => {
  try {
    log.setLevel('info')
    const startNumber = stringToIntegerOrUndefined(process.argv[2])
    const endNumber = stringToIntegerOrUndefined(process.argv[3])
    log.info(`[main] startNumber: ${startNumber}; endNumber: ${endNumber}`)
    if (startNumber && endNumber && (startNumber <= endNumber)) {
      const numbers = R.range(startNumber, endNumber + 1)
      const fileNames = numbers.map(numberToFileName)
      await downloadImages(fileNames, '/rawImages', RAW_IMAGES_FOLDER)
      await downloadImages(fileNames, '/normalisedImages', NORMALISED_IMAGES_FOLDER)
    } else {
      log.error(`[main] the program arguments don't look sensible`)
    }
  } catch (error) {
    log.error(`[main] ERROR: ${error.message}`)
  }
}

main()
