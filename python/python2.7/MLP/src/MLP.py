import numpy as np
import sys

class Net:
	def __init__(self):
		self.data = sys.argv

		# 0 = tanh function
		# 1 = sigmoid function
		# 2 = ReLU function
		# 3 = Leaky ReLU
		self.activationFunction = 0

		# 0 = batch learning
		# 1 = stochastic learning
		self.learningType = 0

		# The number of layers in the neural net
		self.layers = -1

		# The number of neurons in each layer
		self.shape = []

		self.desiredAccuracy = .05

		self.consecErrorReached = 0

		# Learning rate
		self.lr = 0.001

		# Training momentum
		self.momentum = .1

		self.prevDelta = []

		# The error from the last epoch
		self.currentError = -1

		# The number of epochs for the initial training
		self.initialEpoch = 10000

		# The total number of epochs that have passed
		self.totalEpochs = 0

		# The current input for training (only used in stochastic training)
		self.trainingInput = 0

		# Matrix of the all training data inputs
		self.x = []

		# Matrix of answers to training data
		self.y = []

		self.labels = dict()

		# The matrix of weights
		# Each entry is connection between layers
		self.weights = []

		# The matrix of neurons
		# Each entry is a layer of neurons
		self.neurons = []

		# The matrix of biases
		#Each entry is a bias for a layer
		self.biases = []

	########## Global Functions

	# Print information on how to use this program
	def printUsage(self):
		print "USAGE:"
		print "python MLP.py <data file> <activation function> <learning type> <learning rate> <number of training epochs> <input layer shape> <hidden layer shape> ... <output layer shape>\n"
		print "current working activation functions are \"tanh\", \"sigmoid\", \"ReLU\", and \"LReLU\"."
		print "learning type choices are \"batch\" and \"stochastic\".\n"

	#Print the weights and layers of the neural net
	def printNet(self):
		print "\nneurons"
		for layer in range(0, self.layers):
			print self.neurons[layer]

		print "\nweights"
		for layer in range(0, self.layers - 1):
			print self.weights[layer]

		print "\nbiases"
		print self.biases
	
	# Read the data from the given file
	# Initialize the first layer of neurons
	# Initialize the y matrix with training data answers
	# Print an error if the matrix size doesnt match with the data
	# Print an error if there is an error in the training data
	def initTrainingData(self, trainingData):
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

			if (len(data) != (int)(self.data[6])) or (len(answer) != (int)(sys.argv[len(sys.argv) - 1])):
				print "The Training Data File Is Incorrectly Formated\n"
				sys.exit(0)
	
			data = map(float, data)
			temp.append(data)
			
			if self.learningType == 1 and dataPoint == 0:
				temp2.append(data)

			answer = map(float, answer)
			self.y.append(answer)
			self.labels[tuple(data)] = answer

		if self.learningType == 0:
			self.neurons.append(np.array(temp))

		elif self.learningType == 1:
			self.neurons.append(np.array(temp2))

		self.x.append(np.array(temp))

	# Parse Command Line Arguments
	# Initialize the shape of the neural net
	# Print an error if the training data can't be accessed
	# Print an error if there is not enough layers
	def parseArgs(self):
		if len(sys.argv) == 1:
			print "not enough args."
			self.printUsage()
			sys.exit(0)

		try:
			self.trainingData = open(self.data[1], 'r')

		except IOError:
			print "The Given Training Data File, " + str(self.data[1]) + ", Could Not Be Found Or Opened\n"
			self.printUsage()
			sys.exit(0)

		if self.data[2] == "tanh":
			self.activationFunction = 0

		elif self.data[2] == "sigmoid":
			self.activationFunction = 1

		elif self.data[2] == "relu" or self.data[2] == "ReLU":
			self.activationFunction = 2

		elif self.data[2] == "lrelu" or self.data[2] == "LReLU":
			self.activationFunction = 3

		else:
			print str(self.data[2]) + " is not a known activation function.\n"
			self.printUsage()
			sys.exit(0)

		if self.data[3] == "batch":
			self.learningType = 0

		elif self.data[3] == "stochastic":
			self.learningType = 1

		self.lr = float(self.data[4])

		self.initialEpoch = int(self.data[5])

		self.layers = len(self.data) - 6
		if self.layers < 3:
			print "Neural net must have at least 3 layers(input, hidden, output).\n"
			self.printUsage()
			sys.exit(0)

		for layerSize in range(6, len(self.data)):
			self.shape.append((int)(self.data[layerSize]))

		self.initTrainingData(self.trainingData)
		return True

	# Initialized Weight List using a normal distribution
	def initWeights(self):
		for layer in range(0, self.layers - 1):
			layerWeights = np.random.uniform(.01, .1, size = (self.shape[layer], self.shape[layer + 1]))
			self.weights.append(layerWeights)

	# Initialized Neuron list to all 0
	def initNeurons(self):
		for layer in range(1, self.layers):
			neuronLayer = []
			for neuron in range(0, self.shape[layer]):
				neuronLayer.append(0)

			self.neurons.append(np.array(neuronLayer))

	# Initialize Bias List using a normal distribution
	def initBiases(self):
		for layer in range(0, self.layers - 1):
			bias = np.random.uniform(.01, .1, size = None)
			self.biases = np.append(self.biases, bias)

	#Fully Initialized Neural Net
	def initNeuralNet(self):
		self.parseArgs()
		self.initWeights()
		self.initNeurons()
		self.initBiases()

	# Sigmoid Activation Function
	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))

	# Derivative of sigmoid function
	def sigmoidDerivative(self, x):
		return x * (1 - x)

	# tanh Activation Function
	def tanh(self, x):
		return np.tanh(x)

	# Derivative of tanh Function
	def tanhDerivative(self, x):
		return 1 - np.tanh(x) ** 2

	def ReLU(self, x):
		return np.maximum(0, x)

	def ReLUDerivative(self, x):
		return np.greater(x, 0).astype(int)

	def LReLU(self, x):
		x[x < 0] *= .01
		return x

	def LReLUDerivative(self, x):
		x[x < 0] *= .01
		x[x > 0] = (int)(1)
		return x

	def softmax(self, x):
		norm = x - np.max(x)
		return np.exp(norm) / np.sum(np.exp(norm))

	def crossEntropy(self, x, y):
		m = y.shape[0]
		p = softmax(X)
		logLikelihood = -np.log(p[range(m), y])
		loss = np.sum(logLikelihood / m)
		return loss

	def crossEntropyDerivative(self, x, y):
		m = y.shape[0]
		grad = softmax(x)
		grad[range(m), y] -= 1
		grad /= m
		return grad

	# Check to see if the desired accuracy has been achieved
	# To be used for simple linear classification
	def checkOutput(self):
		if np.mean(np.abs(self.currentError)) < self.desiredAccuracy:
			return 1

		return 0

	# Feed Forward Algorithm
	# Calculates the value(s) of the next layer
	def forwardPass(self, inputLayer, inputWeights, layerBias):
		nextLayer = np.dot(inputLayer, inputWeights)
		nextLayer += layerBias
		if self.activationFunction == 0:
			nextLayer = self.tanh(nextLayer)

		elif self.activationFunction == 1:
			nextLayer = self.sigmoid(nextLayer)

		elif self.activationFunction == 2:
			nextLayer = self.ReLU(nextLayer)

		elif self.activationFunction == 3:
			nextLayer = self.LReLU(nextLayer)

		else:
			nextLayer = self.tanh(nextLayer)

		return nextLayer

	# Backpropagation Algorithm
	# Updates the weights and biases feeding the current layer
	def backwardPass(self, layerNum, layer, prevLayer, inputWeights, dOutput, lr):
		if layerNum == self.layers - 1:
			if self.learningType == 0:
				error = self.y - layer

			elif self.learningType == 1:
				error = self.y[self.trainingInput] - layer

			self.currentError = error

			print self.labels.get(tuple(self.x[0][self.trainingInput]))

		else:
			error = dOutput.dot(inputWeights.T)	

		if self.activationFunction == 0:
			slope = self.tanhDerivative(layer)

		elif self.activationFunction == 1:
			slope = self.sigmoidDerivative(layer)

		elif self.activationFunction == 2:
			slope = self.ReLUDerivative(layer)

		elif self.activationFunction == 3:
			slope = self.LReLUDerivative(layer)

		else:
			slope = self.tanhDerivative(layer)

		delta = error * slope
		if not self.prevDelta or self.learningType == 0:
			self.weights[layerNum - 1] += prevLayer.T.dot(delta) * self.lr

		else:
			self.weights[layerNum - 1] += (prevLayer.T.dot(delta) * self.lr) + (self.prevDelta[self.layers - layerNum - 1] * self.momentum)

		self.biases[layerNum - 1] += np.sum(delta) * lr
		return delta

	# Train the neural net on a data set
	def train(self, requestedEpoch):
		epochElapsed = 0
		consecErrorReached = 0

		if requestedEpoch == None:
			self.maxEpoch = self.initialEpoch

		elif requestedEpoch != None or requestedEpoch > 0:
			self.maxEpoch = requestedEpoch

		while True:
			#Forward Propogation
			for layer in range(0, self.layers - 1):
				self.neurons[layer + 1] = self.forwardPass(self.neurons[layer], self.weights[layer], self.biases[layer]) 

			#Check if we reached desired accuracy
			if epochElapsed == self.maxEpoch:
				break

			elif self.learningType == 0 and self.checkOutput() == 1:
				break

			elif self.learningType == 1 and self.checkOutput() == 1:
				consecErrorReached += 1

			elif self.learningType == 1 and self.checkOutput() == 0:
				consecErrorReached = 0

			if consecErrorReached == len(self.x[0]) * 4:
				break

			#Inccrease epoch elapsed
			epochElapsed += 1

			#Print the error every 10000 epochs
			if epochElapsed % 10000 == 0 or epochElapsed == 1:
				print "Error:" + str(np.mean(np.abs(self.currentError)))	

			#Back Propagation
			delta = None
			temp = []
			for layer in range(self.layers - 1, 0, -1):
				inputWeights = None
				if layer != (self.layers - 1):
					inputWeights = self.weights[layer]

				delta = self.backwardPass(layer, self.neurons[layer], self.neurons[layer - 1], inputWeights, delta, self.lr)
				temp.append(delta)

			if self.learningType == 1:
				self.trainingInput = np.random.randint(0, len(self.x[0]), None)
				for inputdatum in range(0, self.shape[0]):
					self.neurons[0][0][inputdatum] = self.x[0][self.trainingInput][inputdatum]

			self.prevDelta = temp

		self.totalEpochs += epochElapsed
		print "\nNumber of epochs in current training session: ",
		print epochElapsed
		print "Total epochs over all training sessions: ",
		print self.totalEpochs

	def run(self):
		while True:
			userCommand = raw_input("\nNeural Net has finished its training. Here is a list of available commands.\n\"predict\"\tNeural Net is ready to process and predict your input.\n\"train\"\t\tNeural Net may be trained further.\n\"print\"\t\tPrint the values of the neural net.\n\"error\"\t\tPrint the current error.\n\"quit\"\t\tQuit the program\n--> ")
			if userCommand == "quit" or userCommand == "exit" or userCommand == "q":
				print "Thank you for using my neural net program.\n"
				sys.exit(0)

			elif userCommand == "predict":
				inputs = []
				for netInput in range(0, len(self.neurons[0][0])):
					inputs.append(input("Please enter the next input datum.\n"))

				self.neurons[1] = self.forwardPass(inputs, self.weights[0], self.biases[0])
				for layer in range(1, self.layers - 1):
					self.neurons[layer + 1] = self.forwardPass(self.neurons[layer], self.weights[layer], self.biases[layer])

				print self.neurons[self.layers - 1]

			elif userCommand == "train":
				print "Previous desired accuracy was " + str(self.desiredAccuracy)
				newAccuracy = input("Please enter a new desired accuracy value.\n")
				self.desiredAccuracy = newAccuracy
				requestedEpochs = input("Please enter the maximum epochs to train over.\n")
				train(requestedEpochs)

			elif userCommand == "print":
				self.printNet()

			elif userCommand == "error":
				print "Expected:" + str(self.y)
				print self.currentError
				print "Error:" + str(np.mean(np.abs(self.currentError)))

			else:
				print "That is not a valid command.\n"

net = Net()
net.initNeuralNet()
net.train(None)
net.run()
