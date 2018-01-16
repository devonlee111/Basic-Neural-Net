import numpy as np
import sys

#Global Variables

y = []	

layers = -1
shape = []
desiredAccuracy = .05
#Learning rate
lr = .05
currentError = -1

weights = []
neurons = []
biases = []

#Print the different weights and layers of the neural net
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

		if (len(data) != (int)(sys.argv[2])) or (len(answer) != (int)(sys.argv[len(sys.argv) - 1])):
			print "The Training Data File Is Incorrectly Formated\n"
			sys.exit(0)
	
		data = map(int, data)
		temp.append(data)
		answer = map(int, answer)
		y.append(answer)

	neurons.append(np.array(temp))
	print y

#Parse Command Line Arguments
def parseArgs():
	global layers
	if len(sys.argv) < 2:
		print "Not Enough Arguments\n"
		sys.exit(0)

	try:
		trainingData = open(sys.argv[1], 'r')
	except IOError:
		print "The Given Training Data File Could Not Be Found Or Opened\n"
		sys.exit(0)

	layers = len(sys.argv) - 2
	if (layers < 3):
		print "Neural Net Must Have at least 3 layers (input, hidden, output).\n"
		sys.exit(0)
	
	for layerSize in range(2, len(sys.argv)):
		shape.append((int)(sys.argv[layerSize]))

	initTrainingData(trainingData)
	return True

#Initialized Weight List 
def initWeights(shape, layers):
	global weights
	for layer in range(0, layers - 1):
		layerWeights = np.random.normal(loc = .5, scale = .01, size = (shape[layer], shape[layer + 1]))
		weights.append(layerWeights)

#Initialized Neurpn List
def initNeurons(shape, layers):
	global neurons
	for layer in range(1, layers):
		neuronLayer = []
		for neuron in range(0, shape[layer]):
			neuronLayer.append(0)
		
		neurons.append(np.array(neuronLayer))

#Initialize Bias List
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

#Sigmoid Activation Function
def sigmoid(x):
	return 1/(1 + np.exp(-x))

#Sigmoid Function Derivative
def sigmoidDerivative(x):
	return x * (1 - x)

#Check to see if the desired accuracy has been achieved
def checkOutput():
	global desiredAccuracy
	if np.mean(np.abs(currentError)) < desiredAccuracy:
		return 1

	return 0

#Feed Forward Algorithm
def forwardPass(inputLayer, inputWeights, layerBias):
	layer = np.dot(inputLayer, inputWeights)
	layer += layerBias
	layer = sigmoid(layer)
	return layer

#Backpropagation Algorithm
def backwardPass(layerNum, layer, prevLayer, inputWeights, dOutput, lr):
	global weights
	global biases
	global layers
	global currentError
	
	if layerNum == layers - 1:
		error = y - layer
		currentError = error
	else:
		error = dOutput.dot(inputWeights.T)
	
	#print layer
	slope = sigmoidDerivative(layer)
	delta = error * slope
	weights[layerNum - 1] += prevLayer.T.dot(delta) * lr
	biases[layerNum - 1] += np.sum(delta) * lr
	
	return delta

def train(y, lr):
	#Initialize variables
	global layers
	global bh
	global inputNeurons
	global hiddenLayerNeurons
	global outputNeurons
	global wh
	global wout
	global bout
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

initNeuralNet()
train(y, lr)
