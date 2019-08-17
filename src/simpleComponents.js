import * as U from './utils'

export const addTrainingSection = (name, onTrain, onSave, onLoad) => {
  const parentElement = document.getElementById('training-sections')
  const template = document.getElementById('training-section-template')
  const clone = document.importNode(template.content, true)
  const div = clone.querySelector('div')
  div.setAttribute('id', `training-section-${name}`)
  const trainingName = clone.querySelector('.training-name')
  const trainBtn = clone.querySelector('.train-btn')
  const saveBtn = clone.querySelector('.save-btn')
  const loadBtn = clone.querySelector('.load-btn')
  trainingName.textContent = U.formatSectionName(name)
  trainBtn.addEventListener('click', onTrain)
  saveBtn.addEventListener('click', onSave(name))
  loadBtn.addEventListener('click', onLoad(name))
  parentElement.appendChild(clone)
}

export const addPredictionSection = (name, onPredict) => {
  const parentElement = document.getElementById('prediction-sections')
  const template = document.getElementById('prediction-section-template')
  const clone = document.importNode(template.content, true)
  const div = clone.querySelector('div')
  div.setAttribute('id', `prediction-section-${name}`)
  const predictionName = clone.querySelector('.prediction-name')
  const predictBtn = clone.querySelector('.predict-btn')
  const clearBtn = clone.querySelector('.clear-btn')
  const results = clone.querySelector('.results')
  predictionName.textContent = U.formatSectionName(name)
  predictBtn.addEventListener('click', onPredict)
  clearBtn.addEventListener('click', () => U.deleteChildren(results))
  parentElement.appendChild(clone)
}
