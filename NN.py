import numpy as np


#Global Variables
# BinaryInput
X = np.array([[0,0],[0,1],[1,0],[1,1]])
	
#Output
#XOR Output
y = np.array([[0],[1],[1],[0]])

#NAND Output
#y = np.array([[1],[1],[1],[0]])

desiredAccuracy = .01
#learning rate
lr = .1
inputNeurons = X.shape[1]
hiddenLayerNeurons = 3
outputNeurons = 1

#weight and bias initialization
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
weights.append(wh)
weights.append(wout)

def initWeights(shape, layers):
	global weights
	for layer in range(0, layers):
		for weight in range(0, shape[layer]):
			weight = np.random.normal(loc = .5, scale = .01, size = 1)
			weights.append(weight)

def initNeurons(shape, layers):
	global neurons
	for layer in range(0, layers):
		neuronLayer = []
		for neuron in range(0, shape[layer]):
			neuronLayer.append(0)
		neurons.append(neuronLayer)

def initBiases(layers):
	global biases
	for layer in range(0, layers):
		bias = np.random.normal(loc = .5, scale = .01, size = 1)
		biases.append(bias)

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
def forwardPass(inputs, weights, bias):
	layer = np.dot(inputs, weights)
	layer += bias
	layer = sigmoid(layer)
	return layer

#Backpropagation Algorithm
def backwardPass(doutput, layer, pervLayer, outputWeights):
	global wh
	global bh
	error = output.dot(outputWeights.T)
	slope = sigmoidDerivative(layer)
	delta = error * slope
	wh += prevLayer.T.dot(delta) * lr
	bh += np.sum(delta) * lr
	return delta

def train(X, y, lr):
	#Initialize variables
	global bh
	global inputNeurons
	global hiddenLayerNeurons
	global outputNeurons
	global wh
	global wout
	global bout
	epochElapsed = 0
	epoch = 1000000

	while True:
		#Forward Propogation
		hiddenLayer = forwardPass(X, wh, bh)
		output = forwardPass(hiddenLayer, wout, bout)

		#Check if we reached desired accuracy
		if checkOutput(output, desiredAccuracy) == 1 or epochElapsed >= epoch:
			break	
		epochElapsed += 1

		#Back Propogation
		E = y-output
		slopeOutputLayer = sigmoidDerivative(output)
		dOutput = E * slopeOutputLayer
		wout += hiddenLayer.T.dot(dOutput) * lr
		bout += np.sum(dOutput) * lr

		#backwardPass(dOutput, hiddenLayer, X, wh)

		hiddenLayerError = dOutput.dot(wout.T)
		slopeHiddenLayer = sigmoidDerivative(hiddenLayer)
		dHiddenLayer = hiddenLayerError * slopeHiddenLayer
		wh += X.T.dot(dHiddenLayer) * lr
		bh += np.sum(dHiddenLayer) * lr

	print epochElapsed
	print output

train(X, y, lr)
