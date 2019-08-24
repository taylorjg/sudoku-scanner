import * as R from 'ramda'
import * as C from './constants'
import * as U from './utils'

export const flattenInitialValues = R.compose(R.unnest, R.map(Array.from))

const digitOrSpace = indexedDigitPredictions => index => {
  const indexedDigitPrediction = indexedDigitPredictions.find(R.propEq('index', index))
  return indexedDigitPrediction
    ? indexedDigitPrediction.digitPrediction.toString()
    : C.SPACE
}

export const indexedDigitPredictionsToInitialValues = indexedDigitPredictions =>
  R.compose(
    R.splitEvery(9),
    U.stringFromChars,
    R.map(digitOrSpace(indexedDigitPredictions))
  )(R.range(0, 81))

export const indexedDigitPredictionstToRows = indexedDigitPredictions =>
  indexedDigitPredictions.map(({ digitPrediction, index }) => ({
    coords: {
      row: Math.trunc(index / 9),
      col: index % 9
    },
    isInitialValue: true,
    value: digitPrediction
  }))
