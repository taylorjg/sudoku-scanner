# Description

I want to create a web app to scan and solve a Sudoku puzzle.
The hard part is scanning the puzzle which is what I am attempting to do in this repo.
My plan is to use [TensorFlow.js](https://www.tensorflow.org/js) to:

* Train a model to recognise the bounding box of a Sudoku puzzle
* Train a model to recognise digits

My hope is that if the bounding box is accurate enough, I can estimate the boundaries of the 81 grid squares.

# TODO

* Train a model to recognise the bounding box of a Sudoku puzzle
    * This is not going very well!
        * Currently, the bounding box prediction seems to be the same for all images
        * I think my training data images are too similar
        * I guess I need images with a variety of size/placement of the grid
        * Maybe the target should be the 4 inner major box lines instead of the bounding box ?
* Train a model to recognise the digits
    * I may move onto this step and go back to the bounding box detection later
    * The plan is to extract images of all the digits from the training images (based on `trainingData[index].boundingBox` and  the corresponding `puzzle.initialValues`)
    * Then, use the images of all the digits to train a model to detect digits 1 through 9
* Try to detect blank grid squares
    * Possibly by simply checking if the majority of pixels are above a threshold greyscale value
    * Or just do this as part of digit recognition ?
* Bring all the parts together    

# Links

* [TensorFlow.js](https://www.tensorflow.org/js)
* sudoku-buster
  * [GitHub](https://github.com/taylorjg/sudoku-buster)
  * [Heroku](https://sudoku-buster.herokuapp.com/)
