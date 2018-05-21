# Basic-Neural-Net
An attempt at understanding and building a basic neural net. Currently written using only python, I plan to expand the number of languages used in the future. The working versions of the program have now been expanded to allow the neural net to be fully customizable and scalable.

# Program Files
**Python:**

**net.py:** The original code used to build the most basic neural net.

**MLP.py**: My custom implementation of a neural net written for python2.7.

**MLP3.py**: My custom implementation of a neural net written for python3.

**MLPV.py**: MLP.py that has been adapted to use a gui for control and visualization

# Programs
## net.py:
### Usage:
	python net.py

## MLP.py:
### Usage:
	python MLP.py <data file> <arg1> <arg2> <arg3> ...
The data file and layers arguments are required.

Use the -h/-help flag for more information on arguments.	

## MLP3.py:
### Usage:
	python3 MLP3.py <data file> <activation function> <learning type> <learning rate> <training epochs> <input layer size> <hidden layer 1 size> ... <output layer size>

### Arguments (MLP & MLP3.py):
**Activation functions**: tanh, sigmoid, relu, and lrelu.

**Learning Types**: batch, minibatch, stochastic.

**Input Layer Size**: The number of data points for an individual point of training data.

**Output Layer Size**: Total number of distinct classifications for the training data.

## MLPV.py:
### Usage:
Run the run.sh script.
	
	./run.sh

### Known Bugs:

When given large numbers of neurons, display does not correctly draw location of neurons.

# Data File Structure
The data file should be structured as such:

Line 1 should contain a single number representing how many data sets to include for training.

All following lines should contain input values seperated by commas, followed by a colon, followed by the correct label

**Example (XOR Function):**
	
	4
	0,0:0
	0,1:1
	1,0:1
	1,1:0

# Helper Programs/Scripts
## image_processor.py:
Python2.7 script to convert images to the file format accepted by my neural net.

### Usage:
	python image_processor.py <input file/directory> <output file>

### Notes:
When given a single image path, it will generate a file that can be used to test the net's predictive capability after training. When given a directory path, it will generate a file that can be used to train the net.

When given a path to a directory, the directory should contain sub-directories that are filled with images of the same size/resolution. The name of the sub-directory the image is in will be the label associated with the image.

# Sources
### Websites I used for learning and references:
**net.py original code (this is what I used to intially learn about building a neural net):**

https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/

**Other useful sights for understanding and learning about neural net basics:**

http://iamtrask.github.io/2015/07/12/basic-python-network/

https://beckernick.github.io/neural-network-scratch/

http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

https://ayearofai.com/rohan-4-the-vanishing-gradient-problem-ec68f76ffb9b

https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions

https://machinelearningmastery.com/neural-networks-crash-course/

http://thelaziestprogrammer.com/sharrington/math-of-machine-learning/the-gradient-a-visual-descent

https://jamesmccaffrey.wordpress.com/2017/06/06/neural-network-momentum/
