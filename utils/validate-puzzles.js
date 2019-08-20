const log = require('loglevel')
const puzzles = require('../data/puzzles.json')
const { solve } = require('./logic')

const validatePuzzle = puzzle => {
  log.info(`Validating ${puzzle.description}`)
  const solutions = solve(puzzle.initialValues)
  if (solutions.length !== 1) {
    log.error(`\tsolutions.length: ${solutions.length}`)
  }
}

const main = () => {
  log.setLevel('info')
  for (const puzzle of puzzles) {
    validatePuzzle(puzzle)
  }
}

main()
