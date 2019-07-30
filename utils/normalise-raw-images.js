const { spawn } = require('child_process')
const path = require('path')
const R = require('ramda')

const RAW_IMAGES_FOLDER = path.resolve(__dirname, '..', 'scanned-images', 'raw')
const NORMALISED_IMAGES_FOLDER = path.resolve(__dirname, '..', 'scanned-images', 'normalised')
const PNG_EXT = '.png'

const stringToIntegerOrUndefined = s => {
  const n = Number(s)
  return Number.isInteger(n) ? n : undefined
}

const numberToFileName = n =>
  `${n.toString().padStart(5, 0)}${PNG_EXT}`

const main = async () => {
  try {
    const startNumber = stringToIntegerOrUndefined(process.argv[2])
    const endNumber = stringToIntegerOrUndefined(process.argv[3])
    console.log(`[main] startNumber: ${startNumber}; endNumber: ${endNumber}`)
    if (startNumber && endNumber && (startNumber <= endNumber)) {
      const fileNames =
        R.range(startNumber, endNumber + 1)
          .map(numberToFileName)
          .map(fileName => ({
            input: path.resolve(RAW_IMAGES_FOLDER, fileName),
            output: path.resolve(NORMALISED_IMAGES_FOLDER, fileName)
          }))
      const childProcesses = fileNames.map(({ input, output }) => {
        console.log(`Processing ${input}`)
        const args = [
          'convert',
          input,
          '-resize', '224',
          '-colorspace', 'Gray',
          '-alpha', 'off',
          '-strip',
          output
        ]
        return spawn('magick', args)
      })
      const promises = childProcesses.map(childProcess =>
        new Promise((resolve, reject) => {
          childProcess.on('exit', e => resolve({ code: e.code, signal: e.signal }))
          childProcess.on('error', e => reject(e.message))
        })
      )
      await Promise.all(promises)
    } else {
      console.log(`[main] the program arguments don't look sensible`)
    }
  } catch (error) {
    console.log(`[main] ERROR: ${error.message}`)
  }
}

main()
