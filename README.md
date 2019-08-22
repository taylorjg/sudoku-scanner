# Description

I want to create a web app to scan and solve a Sudoku puzzle.
Apps already exist for Android and iOS but I haven't seen this done as a web app before.
The hard part is scanning the puzzle which is what I am attempting to do in this repo.
My plan is to use [TensorFlow.js](https://www.tensorflow.org/js) to:

* Train a model to find the bounding box of a Sudoku puzzle
* Calculate the grid squares from the bounding box
* Train a model to distinguish between blank grid squares and digits
* Train a model to recognise digits 1-9

> **UPDATE**
> I have been unable to train a model to find the bounding box of a Sudoko puzzle.
> Therefore, I have reluctantly decided to follow the OpenCV route.
> I say reluctantly because OpenCV.js is massive.
> I borrowed heavily from the following links to get something working using OpenCV.js:
>
> * [Emaraic - Real-time Sudoku Solver](http://emaraic.com/blog/realtime-sudoku-solver)
> * [tahaemara/real-time-sudoku-solver: Real-time Sudoku Solver using Opencv and Deeplearning4j](https://github.com/tahaemara/real-time-sudoku-solver)

# Training Failure

_TODO: describe the problem that I encountered_

# Further Work

_TODO: describe plans to have another go at training a model to find the bounding box_

# Instructions

* Open https://sudoku-scanner.herokuapp.com/
* Click 'Load' under 'Training - Blanks' to load a pre-trained model to distinguish between blank grid squares and digits
* Click 'Load' under 'Training - Digits' to load a pre-trained model to recognise digits 1-9
* Click 'Start Camera'
* Click 'Capture Camera' to take a photo of a Sudoku puzzle from a newpaper or similar
 * Try to get the puzzle grid to:
   * roughly fill the guide lines
   * be in focus
   * not be wonky
* Click 'Predict Capture' to scan and solve the puzzle

# TODO

These are some of my major TODO items:

* Instead of using a trial and error process of capturing images and trying to scan them, automatically
try to scan an image from the video stream and stop the camera once a convincing match has been found
* Take the results of this repo and feed them into [sudoku-buster](https://github.com/taylorjg/sudoku-buster)
  * The idea is for `sudoku-buster` to be a polished Sudoku scanning/solving project whereas this repo is a laboratory experiment
* Persevere with trying to train a model to find the bounding box of a Sudoku puzzle so that everything is done via TensorFlow.js and I no longer need to bring in OpenCV.js (which is massive)

# Links

* Tensorflow.js
  * [TensorFlow.js](https://www.tensorflow.org/js)
  * [TensorFlow.js API Reference](https://js.tensorflow.org/api/1.2.7/)
* OpenCV
  * [OpenCV](https://opencv.org/)
  * [OpenCV: OpenCV.js Tutorials](https://docs.opencv.org/3.4/d5/d10/tutorial_js_root.html)
  * [OpenCV: OpenCV modules](https://docs.opencv.org/4.1.1/)
* sudoku-buster
  * [GitHub](https://github.com/taylorjg/sudoku-buster)
  * [Heroku](https://sudoku-buster.herokuapp.com/)
