import numpy as np

#Input array
#X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

# BinaryInput
X = np.array([[0,0],[0,1],[1,0],[1,1]])
print X
	
#Output
#y = np.array([[1],[1],[0]])

#XOR Output
y = np.array([[0],[1],[1],[0]])
print y
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

def forwardPass(layer):
	return layer

def backwardPass(layer):
	return layer

#Variable initialization
epoch = 10000000
epochElapsed = 0
desiredAccuracy = .01
lr = 0.1 #learning rate
inputNeurons = X.shape[1]
hiddenLayerNeurons = 3
outputNeurons = 1

#weight and bias initialization
#Hidden neuron weights
wh=np.random.uniform(size=(inputNeurons, hiddenLayerNeurons))

#Hidden bias weight
#bh=np.random.uniform(size=(1, hiddenLayerNeurons))
bh = .5

#Out weight
wout=np.random.uniform(size=(hiddenLayerNeurons, outputNeurons))

#out bias
bout=np.random.uniform(size=(1, outputNeurons))

#for i in range(epoch):
while True:
	#Forward Propogation
	hiddenLayer = np.dot(X, wh)
	#print "Hidden Layer \n"
	#print hiddenLayer
	#print '\n'
	#print bh
	#print '\n'
	hiddenLayer = hiddenLayer + bh
	#print hiddenLayer
	#print '\n'
	hiddenLayer = sigmoid(hiddenLayer)
	#print hiddenLayer
	#print '\n'	
	
	#print "Output Layer \n"
	outputLayer = np.dot(hiddenLayer, wout)
	#print outputLayer
	#print '\n'
	outputLayer = outputLayer + bout
	#print outputLayer 
	#print '\n'
	output = sigmoid(outputLayer)
	#print outputLayer
	#print '\n'
	
	if checkOutput(output, desiredAccuracy) == 1 or epochElapsed >= epoch:
		break	
	epochElapsed += 1

	#Backpropagation
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
