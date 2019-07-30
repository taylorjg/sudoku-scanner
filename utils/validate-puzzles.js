const puzzles = require('../data/puzzles.json')
const { solve } = require('./logic')

const validatePuzzle = puzzle => {
  console.log(`Validating ${puzzle.description}`)
  const solutions = solve(puzzle.initialValues)
  if (solutions.length !== 1) {
    console.log(`\tERROR: solutions.length: ${solutions.length}`)
  }
}

const main = () => {
  for (const puzzle of puzzles) {
    validatePuzzle(puzzle)
  }
}

main()
