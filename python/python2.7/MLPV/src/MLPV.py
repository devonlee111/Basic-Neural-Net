import numpy as np
import sys

class Net:
	##########Global Variables

	# 0 = tanh function
	# 1 = sigmoid function
	# 2 = ReLU function
	# 3 = Leaky ReLU
	activationFunction = 0

	# 0 = batch learning
	# 1 = stochastic learning
	learningType = 0

	# The number of layers in the neural net
	layers = -1

	# The number of neurons in each layer
	shape = []

	desiredAccuracy = .05

	consecErrorReached = 0

	# Learning rate
	lr = 0.001

	# Training momentum
	momentum = .1

	prevDelta = []

	# The error from the last epoch
	currentError = -1

	# The number of epochs for the initial training
	initialEpoch = 100000

	# The total number of epochs that have passed
	totalEpochs = 0

	# The current input for training (only used in stochastic training)
	trainingInput = 0

	# Matrix of the all training data inputs
	x = []

	# Matrix of answers to training data
	y = []

	# The matrix of weights
	# Each entry is connection between layers
	weights = []

	# The matrix of neurons
	# Each entry is a layer of neurons
	neurons = []

	# The matrix of biases
	# Each entry is a bias for a layer
	biases = []

	##########Functions

	# Read the data from the given file
	# Initialize the first layer of neurons
	# Initialize the y matrix with training data answers
	# Print an error if the matrix size doesnt match with the data
	# Print an error if there is an error in the training data
	def initTrainingData(trainingData):
		global neurons
		global x

		dataSize = (int)(trainingData.readline())
		temp = []
		temp2 = []
		for dataPoint in range(0, dataSize):
			line = trainingData.readline()
			line = line.rstrip("\n")
			if not line:
				print "The Training Data File Is Incorrectly Formated\n"
				sys.exit(0)
	
			answer = line.split(":")	
			data = answer[0].split(",")
			answer = answer[1].split(",")

			if (len(data) != (int)(sys.argv[6])) or (len(answer) != (int)(sys.argv[len(sys.argv) - 1])):
				print "The Training Data File Is Incorrectly Formated\n"
				sys.exit(0)
	
			data = map(float, data)
			temp.append(data)
			if learningType == 1 and dataPoint == 0:
				temp2.append(data)

			answer = map(float, answer)
			y.append(answer)

		if learningType == 0:
			neurons.append(np.array(temp))

		elif learningType == 1:
			neurons.append(np.array(temp2))

		x.append(np.array(temp))

	# Initialized Weight List using a normal distribution
	def initWeights():
		global weights

		for layer in range(0, layers - 1):
			layerWeights = np.random.uniform(.01, .1, size = (shape[layer], shape[layer + 1]))
			weights.append(layerWeights)

	# Initialized Neuron list to all 0
	def initNeurons():
		global neurons

		for layer in range(1, layers):
			neuronLayer = []
			for neuron in range(0, shape[layer]):
				neuronLayer.append(0)

			neurons.append(np.array(neuronLayer))

	# Initialize Bias List using a normal distribution
	def initBiases():
		global biases

		for layer in range(0, layers - 1):
			bias = np.random.uniform(.01, .1, size = None)
			biases = np.append(biases, bias)

	#Fully Initialized Neural Net
	def initNeuralNet():
		parseArgs()
		initWeights()
		initNeurons()
		initBiases()

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

	def ReLU(x):
		return np.maximum(0, x)

	def ReLUDerivative(x):
		return np.greater(x, 0).astype(int)

	def LReLU(x):
		x[x < 0] *= .01
		return x

	def LReLUDerivative(x):
		x[x < 0] *= .01
		x[x > 0] = (int)(1)
		return x

	def softmax(x):
		return np.exp(x) / float(sum(np.exp(x)))

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

		elif activationFunction == 2:
			nextLayer = ReLU(nextLayer)

		elif activationFunction == 3:
			nextLayer = LReLU(nextLayer)

		else:
			nextLayer = tanh(nextLayer)

		return nextLayer

	# Backpropagation Algorithm
	# Updates the weights and biases feeding the current layer
	def backwardPass(layerNum, layer, prevLayer, inputWeights, dOutput, lr):
		global weights
		global biases
		global currentError

		if layerNum == layers - 1:
			if learningType == 0:
				error = y - layer

			elif learningType == 1:
				error = y[trainingInput] - layer

			currentError = error

		else:
			error = dOutput.dot(inputWeights.T)	

		if activationFunction == 0:
			slope = tanhDerivative(layer)

		elif activationFunction == 1:
			slope = sigmoidDerivative(layer)

		elif activationFunction == 2:
			slope = ReLUDerivative(layer)

		elif activationFunction == 3:
			slope = LReLUDerivative(layer)

		else:
			slope = tanhDerivative(layer)

		delta = error * slope
		if not prevDelta or learningType == 0:
			weights[layerNum - 1] += prevLayer.T.dot(delta) * lr

		else:
			weights[layerNum - 1] += (prevLayer.T.dot(delta) * lr) + (prevDelta[layers - layerNum - 1] * momentum)

		biases[layerNum - 1] += np.sum(delta) * lr
		return delta

	# Train the neural net on a data set
	def train(requestedEpoch):
		global neurons
		global totalEpochs
		global trainingInput
		global consecErrorReached
		global prevDelta

		consecErrorReached = 0
		epochElapsed = 0

		if requestedEpoch == None:
			maxEpoch = initialEpoch

		elif requestedEpoch != None or requestedEpoch > 0:
			maxEpoch = requestedEpoch

		while True:
			#Forward Propogation
			for layer in range(0, layers - 1):
				neurons[layer + 1] = forwardPass(neurons[layer], weights[layer], biases[layer]) 

			#Check if we reached desired accuracy
			if epochElapsed == maxEpoch:
				break

			elif learningType == 0 and checkOutput() == 1:
				break

			elif learningType == 1 and checkOutput() == 1:
				consecErrorReached += 1

			elif learningType == 1 and checkOutput() == 0:
				consecErrorReached = 0

			if consecErrorReached == len(x[0]) * 4:
				break

			#Inccrease epoch elapsed
			epochElapsed += 1

			#Print the error every 10000 epochs
			if epochElapsed % 10000 == 0 or epochElapsed == 1:
				print "Error:" + str(np.mean(np.abs(currentError)))	

			#Back Propagation
			delta = None
			temp = []
			for layer in range(layers - 1, 0, -1):
				inputWeights = None
				if layer != (layers - 1):
					inputWeights = weights[layer]

				delta = backwardPass(layer, neurons[layer], neurons[layer - 1], inputWeights, delta, lr)
				temp.append(delta)

			if learningType == 1:
				trainingInput = np.random.randint(0, len(x[0]), None)
				for inputdatum in range(0, shape[0]):
					neurons[0][0][inputdatum] = x[0][trainingInput][inputdatum]

			prevDelta = temp

		totalEpochs += epochElapsed
		print "\nNumber of epochs in current training session: ",
		print epochElapsed
		print "Total epochs over all training sessions: ",
		print totalEpochs
