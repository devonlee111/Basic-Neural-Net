import numpy as np

# BinaryInput
X = np.array([[0,0],[0,1],[1,0],[1,1]])
	
#Output
#XOR Output
y = np.array([[0],[1],[1],[0]])

#NAND Output
#y = np.array([[1],[1],[1],[0]])


def checkOutput(output, accuracy):
	if abs(output[0] - y[0]) < accuracy and abs(output[1] - y[1]) < accuracy and abs(output[2] - y[2]) < accuracy and abs(output[3] - y[3]) < accuracy:
		return 1

	return 0

#Sigmoid Function
def sigmoid(x):
	return 1/(1 + np.exp(-x))

#Derivative of Sigmoid
def sigmoidDerivative(x):
	return x * (1 - x)

#Feed Forward Algorithm
def forwardPass(inputs, weights, bias):
	layer = np.dot(inputs, weights)
	layer += bias
	layer = sigmoid(layer)
	return layer

#Back Propagation Algorithm
def backwardPass(layer):
	return layer

#Variable initialization
desiredAccuracy = .01
lr = 0.1 #learning rate
inputNeurons = X.shape[1]
hiddenLayerNeurons = 3
outputNeurons = 1

#weight and bias initialization
#Hidden neuron weights
wh = np.random.normal(loc = .5, scale = .01, size = (inputNeurons, hiddenLayerNeurons))

#Hidden bias weight
bh = .5

#Out weight
wout = np.random.normal(loc = .5, scale = .01, size = (hiddenLayerNeurons, outputNeurons))

#out bias
bout = np.random.normal(loc = .5, scale = .01, size = (1, outputNeurons))

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
		bout += np.sum(dOutput, axis=0, keepdims=True) * lr

		hiddenLayerError = dOutput.dot(wout.T)
		slopeHiddenLayer = sigmoidDerivative(hiddenLayer)
		dHiddenLayer = hiddenLayerError * slopeHiddenLayer
		wh += X.T.dot(dHiddenLayer) * lr
		bh += np.sum(dHiddenLayer) * lr

	print epochElapsed
	print output

train(X, y, lr)
