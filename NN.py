import numpy as np
import sys

#Global Variables
# BinaryInput
X = np.array([[0,0],[0,1],[1,0],[1,1]])
	
#Output
#XOR Output
y = np.array([[0],[1],[1],[0]])

#NAND Output
#y = np.array([[1],[1],[1],[0]])

layers = -1
shape = []
desiredAccuracy = .05
#Learning rate
lr = 1
inputNeurons = X.shape[1]
hiddenLayerNeurons = 3
outputNeurons = 1

#Weight and bias initialization
#Hidden neuron weights
wh = np.random.normal(loc = .5, scale = .01, size = (inputNeurons, hiddenLayerNeurons))

#Hidden bias weight
bh = np.random.normal(loc = .5, scale = .01, size = 1)

#Out weight
wout = np.random.normal(loc = .5, scale = .01, size = (hiddenLayerNeurons, outputNeurons))

#out bias
bout = np.random.normal(loc = .5, scale = .01, size = 1)

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
		
		data = line.split(",")
		if len(data) != (int)(sys.argv[3]):
			print "The Training Data File Is Incorrectly Formated\n"
			sys.exit(0)
		
		data = map(int, data)
		temp.append(data)

	neurons.append(np.array(temp))

#Parse Command Line Arguments
def parseArgs():
	global layers
	if len(sys.argv) < 3:
		print "Not Enough Arguments\n"
		sys.exit(0)

	try:
		trainingData = open(sys.argv[1], 'r')
	except IOError:
		print "The Given Training Data File Could Not Be Found Or Opened\n"
		sys.exit(0)

	layers = (int)(sys.argv[2])
	if (layers < 3):
		print "Neural Net Must Have at least 3 layers (input, hidden, output).\n"
		sys.exit(0)
	
	if (len(sys.argv) - 3) != layers:
		print "Invalid Neural Network Dimensions"
		sys.exit(0)

	for layerSize in range(3, len(sys.argv)):
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
def checkOutput(output, accuracy):
	if abs(output[0] - y[0]) < accuracy and abs(output[1] - y[1]) < accuracy and abs(output[2] - y[2]) < accuracy and abs(output[3] - y[3]) < accuracy:
		return 1

	return 0

#Feed Forward Algorithm
def forwardPass(inputLayer, inputWeights, layerBias):
	layer = np.dot(inputLayer, inputWeights)
	layer += layerBias
	layer = sigmoid(layer)
	return layer

#Backpropagation Algorithm
def backwardPass(layerNum, layer, prevLayer, outputWeights, dOutput, lr):
	global weights
	global biases
	global layers

	if layerNum == layers - 1:
		error = y - layer
	else:
		error = dOutput.dot(outputWeights.T)

	slope = sigmoidDerivative(layer)
	delta = error * slope
	weights[layerNum - 1] += prevLayer.T.dot(delta) * lr
	biases[layerNum - 1] += np.sum(delta) * lr
	
	#wh += prevLayer.T.dot(delta) * lr
	#bh += np.sum(delta) * lr
	return delta

def train(X, y, lr):
	#Initialize variables
	global layers
	global bh
	global inputNeurons
	global hiddenLayerNeurons
	global outputNeurons
	global wh
	global wout
	global bout
	
	global neurons
	global weights
	global biases

	epochElapsed = 0
	maxEpoch = 500000

	while True:
		#Forward Propogation
		#hiddenLayer = forwardPass(X, wh, bh)
		#output = forwardPass(hiddenLayer, wout, bout)
		
		for layer in range(0, layers - 1):
			neurons[layer + 1] = forwardPass(neurons[layer], weights[layer], biases[layer]) 

		#Check if we reached desired accuracy
		if checkOutput(neurons[layers - 1], desiredAccuracy) == 1 or epochElapsed >= maxEpoch:
			break

		#Increase the number of elapsed epochs
		epochElapsed += 1

		#Back Propogation
		#E = y-output
		#slopeOutputLayer = sigmoidDerivative(output)
		#dOutput = E * slopeOutputLayer
		#wout += hiddenLayer.T.dot(dOutput) * lr
		#bout += np.sum(dOutput) * lr

		#hiddenLayerError = dOutput.dot(wout.T)
		#slopeHiddenLayer = sigmoidDerivative(hiddenLayer)
		#dHiddenLayer = hiddenLayerError * slopeHiddenLayer
		#wh += X.T.dot(dHiddenLayer) * lr
		#bh += np.sum(dHiddenLayer) * lr
		
		dOutput = None
		for layer in range(layers - 1, 0, -1):
			if layer == layers - 1:
				error = y - neurons[layer]
				dOutput = backwardPass(layer, neurons[layer], neurons[layer - 1], None, dOutput, lr)
			else:
				dOutput = backwardPass(layer, neurons[layer], neurons[layer - 1], weights[layer], dOutput, lr)

	print "epoch: ",
	print epochElapsed
	print neurons[layers - 1]

initNeuralNet()
train(X,y, lr)
