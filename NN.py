import numpy as np
import sys

##########Global Variables

# 0 = tanh function
# 1 = sigmoid function
activationFunction = 0

# The number of layers in the neural net
layers = -1

# The number of neurons in each layer
shape = []

desiredAccuracy = .05

# Learning rate
lr = .01

currentError = -1

# Matrix of answers to training data
y = []

# The matrix of weights
# Each entry is connection between layers
weights = []

# The matrix of neurons
# Each entry is a layer of neurons
neurons = []

# The matrix of biases
#Each entry is a bias for a layer
biases = []

##########Functions

#Print the weights and layers of the neural net
def printNet():
	global neurons
	global weights
	global biases
	global layers
	global wh
	global wout

	print "\nneurons"
	for layer in range(0, layers):
		print neurons[layer]

	print "\nweights"
	for layer in range(0, layers - 1):
		print weights[layer]

	print "\nbiases"
	print biases
	print "\n"
	
# Read the data from the given file
# Initialize the first layer of neurons
# Initialize the y matrix with training data answers
# Print an error if the matrix size doesnt match with the data
# Print an error if there is an error in the training data
def initTrainingData(trainingData):
	global neurons

	dataSize = (int)(trainingData.readline())
	temp = []
	for dataPoint in range(0, dataSize):
		line = trainingData.readline()
		line = line.rstrip("\n")
		if not line:
			print "The Training Data File Is Incorrectly Formated\n"
			sys.exit(0)
	
		answer = line.split(":")	
		data = answer[0].split(",")
		answer = answer[1].split(",")

		if (len(data) != (int)(sys.argv[3])) or (len(answer) != (int)(sys.argv[len(sys.argv) - 1])):
			print "The Training Data File Is Incorrectly Formated\n"
			sys.exit(0)
	
		data = map(float, data)
		temp.append(data)
		answer = map(float, answer)
		y.append(answer)

	neurons.append(np.array(temp))
	print y

# Parse Command Line Arguments
# Initialize the shape of the neural net
# Print an error if the training data can't be accessed
# Print an error if there is not enough layers
def parseArgs():
	global layers
	global activationFunction

	if len(sys.argv) < 3:
		print "Not Enough Arguments\n"
		sys.exit(0)

	try:
		trainingData = open(sys.argv[1], 'r')

	except IOError:
		print "The Given Training Data File Could Not Be Found Or Opened\n"
		sys.exit(0)

	layers = len(sys.argv) - 3
	if (layers < 3):
		print "Neural Net Must Have at least 3 layers (input, hidden, output).\n"
		sys.exit(0)

	if (sys.argv[2] == "tanh"):
		activationFunction = 0
	elif (sys.argv[2] == "sigmoid"):
		activationFunction = 1
	
	for layerSize in range(3, len(sys.argv)):
		shape.append((int)(sys.argv[layerSize]))

	initTrainingData(trainingData)
	return True

# Initialized Weight List using a normal distribution
def initWeights(shape, layers):
	global weights

	for layer in range(0, layers - 1):
		layerWeights = np.random.normal(loc = .5, scale = .01, size = (shape[layer], shape[layer + 1]))
		weights.append(layerWeights)

# Initialized Neuron List using a normal distribution
def initNeurons(shape, layers):
	global neurons
	for layer in range(1, layers):
		neuronLayer = []
		for neuron in range(0, shape[layer]):
			neuronLayer.append(0)
		
		neurons.append(np.array(neuronLayer))

# Initialize Bias List using a normal distribution
def initBiases(layers):
	global biases

	for layer in range(0, layers - 1):
		bias = np.random.normal(loc = .5, scale = .01, size = None)
		biases = np.append(biases, bias)

#Fully Initialized Neural Net
def initNeuralNet():
	global layers
	global shape
	
	parseArgs()
	initWeights(shape, layers)
	initNeurons(shape, layers)
	initBiases(layers)

# Sigmoid Activation Function
def sigmoid(x):
	return 1/(1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoidDerivative(x):
	return x * (1 - x)

# tanh Activation Function
def tanh(x):
	return np.tanh(x)

# Derivative of tanh Function
def tanhDerivative(x):
	return 1 - np.tanh(x) ** 2

#Check to see if the desired accuracy has been achieved
def checkOutput():
	global desiredAccuracy
	if np.mean(np.abs(currentError)) < desiredAccuracy:
		return 1

	return 0

# Feed Forward Algorithm
# Calculates the value(s) of the next layer
def forwardPass(inputLayer, inputWeights, layerBias):
	global activationFunction

	nextLayer = np.dot(inputLayer, inputWeights)
	nextLayer += layerBias
	if activationFunction == 0:
		nextLayer = tanh(nextLayer)
	elif activationFunction == 1:
		nextLayer = sigmoid(nextLayer)
	else:
		nextLayer = tanh(nextLayer)

	return nextLayer

# Backpropagation Algorithm
# Updates the weights and biases feeding the current layer
def backwardPass(layerNum, layer, prevLayer, inputWeights, dOutput, lr):
	global weights
	global biases
	global layers
	global currentError
	global activationFunction

	if layerNum == layers - 1:
		error = y - layer
		currentError = error
	else:
		error = dOutput.dot(inputWeights.T)	

	if activationFunction == 0:
		slope = tanhDerivative(layer)
	elif activationFunction == 1:
		slope = sigmoidDerivative(layer)
	else:
		slope = tanhDerivative(layer)

	delta = error * slope
	weights[layerNum - 1] += prevLayer.T.dot(delta) * lr
	biases[layerNum - 1] += np.sum(delta) * lr
	return delta

# Train the neural net on a data set
def train(y, lr):
	global layers
	global currentError
	global neurons
	global weights
	global biases

	epochElapsed = 0	
	maxEpoch = 500000

	while True:
		#Forward Propogation
	
		for layer in range(0, layers - 1):
			neurons[layer + 1] = forwardPass(neurons[layer], weights[layer], biases[layer]) 

		#Check if we reached desired accuracy
		if checkOutput() == 1 or epochElapsed >= maxEpoch:
			break

		#Increase the number of elapsed epochs
		epochElapsed += 1

		#Print the error every 10000 epochs
		if epochElapsed % 10000 == 0:
			print "Error:" + str(np.mean(np.abs(currentError)))

		#Back Propagation
		delta = None
		for layer in range(layers - 1, 0, -1):
			if layer == layers - 1:
				delta = backwardPass(layer, neurons[layer], neurons[layer - 1], None, delta, lr)
			else:
				delta = backwardPass(layer, neurons[layer], neurons[layer - 1], weights[layer], delta, lr)

	print "epoch: ",
	print epochElapsed
	print neurons[layers - 1]

def run():
	global neurons
	global weights
	global biases
	global layers

	while True:
		userCommand = raw_input("Neural Net is ready to process your input(p). You may also quit the program(q)\n")
		if userCommand == "q":
			print "Thank you for using my neural net program.\n"
			sys.exit(0)

		elif userCommand == "p":
			inputs = []
			for netInput in range(0, len(neurons[0][0])):
				inputs.append(input("Please enter the next input datum.\n"))

			neurons[1] = forwardPass(inputs, weights[0], biases[0])
			for layer in range(1, layers - 1):
				neurons[layer + 1] = forwardPass(neurons[layer], weights[layer], biases[layer])

			print neurons[layers - 1]

		else:
			print "That is not a valid command.\n"

initNeuralNet()
train(y, lr)
run()
