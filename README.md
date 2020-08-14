# Digit Recognizer
This script was mostly made by following [this handy tutorial](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/). The `model.py` script is what was used to evaluate and train various models. I ended up going with one different than the tutorial as it achieved a higher accuracy on the test sets on average (~99%). `drawing_evaluator.py` is something I added to put the model to the test. It takes user input through a cv2 drawing and runs the model on it.

## How to Use
`python3 drawing_evaluator.py`    
- A window should pop up, draw digits on this window
- Press `Enter` on your keyboard for window to clear and for the console to print out what the model thinks you drew
- Press `Esc` to quit script

## Note
The scripts assume you have Keras, Tensorflow (>=2.2), numpy, and cv2 installed
