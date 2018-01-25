# Basic-Neural-Net
An attempt at understanding and building a basic neural net. Currently written using only python, I plan to expand the number of languages used in the future.

# Program Files
Python:

net.py:	The original code used to build the most basic neural net.

MLP.py:	My custom implementation of a multilayer perceptron using python2.7.

MLP3.py: My custom implementation of a multilayer perceptron using python3.

# Usage
net.py:

	python net.py

MLP.py:

	python MLP.py <data file> <activation function> <learning type> <learning rate> <training epochs> <input layer size> <hidden layer 1 size> ... <output layer size>

MLP3.py:

	python3 MLP3.py <data file> <activation function> <[-b|-s]> <input layer size> <hidden layer 1 size> ... <output layer size>

# Data File Structure
The data file should pe structured as such.

Line 1 should contain a single number representing how many data sets to train over.

All following lines should contain input values seperated by commas, followed by correct output values seperated by commas.

Example:
	
	4
	0,0:0
	0,1:1
	1,0:1
	1,1:0

# Sources
net.py original code and educational site: https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/

Other useful sights:

http://iamtrask.github.io/2015/07/12/basic-python-network/
