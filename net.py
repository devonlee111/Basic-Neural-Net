import numpy as np

#Input array
#X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

# BinaryInput
X = np.array([[0,0],[0,1],[1,0],[1,1]])

#Output
#y = np.array([[1],[1],[0]])

#XOR Output
y = np.array([[0],[1],[1],[0]])

#NAND Output
#y = np.array([[1],[1],[1],[0]])

#Sigmoid Function
def sigmoid (x):
	return 1/(1 + np.exp(-x))

#Derivative of Sigmoid
def sigmoidDerivative(x):
	return x * (1 - x);

#Variable initialization
epoch = 10000
lr = 0.1 #learning rate
inputNeurons = X.shape[1]
hiddenLayerNeurons = 3
outputNeurons = 1

#weight and bias initialization
#Hidden neuron weights
wh=np.random.uniform(size=(inputNeurons, hiddenLayerNeurons))
#Hidden bias weight
bh=np.random.uniform(size=(1, hiddenLayerNeurons))
#Out weight
wout=np.random.uniform(size=(hiddenLayerNeurons, outputNeurons))
#out bias
bout=np.random.uniform(size=(1, outputNeurons))

for i in range(epoch):
	#Forward Propogation
	hiddenLayerInput1 = np.dot(X, wh)
	hiddenLayerInput=hiddenLayerInput1 + bh
	hiddenLayerActivations = sigmoid(hiddenLayerInput)
	outputLayerInput1 = np.dot(hiddenLayerActivations, wout)
	outputLayerInput = outputLayerInput1 + bout
	output = sigmoid(outputLayerInput)

	#Backpropagation
	E = y-output
	slopeOutputLayer = sigmoidDerivative(output)
	slopeHiddenLayer = sigmoidDerivative(hiddenLayerActivations)
	dOutput = E * slopeOutputLayer
	hiddenLayerError = dOutput.dot(wout.T)
	dHiddenLayer = hiddenLayerError * slopeHiddenLayer
	wout += hiddenLayerActivations.T.dot(dOutput) * lr
	bout += np.sum(dOutput, axis=0, keepdims=True) * lr
	wh += X.T.dot(dHiddenLayer) * lr
	bh += np.sum(dHiddenLayer) * lr

print output
