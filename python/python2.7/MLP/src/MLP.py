import numpy as np
import sys

class Net:
	def __init__(self):
		self.data = sys.argv

		# The activation function to use for each layer
		# 0 = tanh function
		# 1 = sigmoid function
		# 2 = ReLU function
		# 3 = Leaky ReLU
		self.activationFunction = 0

		# How we should train the neural net on thr data
		# 0 = batch learning
		# 1 = stochastic learning
		self.learningType = 0

		self.x = []			# Matrix of the all training data inputs
		self.y = []			# one hot Matrix of answers to training data
		self.labels = dict()		# Dictionary linking index of one hot value to answer
		self.layers = 0			# The number of layers in the neural net
		self.shape = []			# The number of neurons in each layer
		self.lr = 0.001			# The learning rate of the neural network
		self.desiredAccuracy = .05	# The threshhold accuracy to stop training
		self.momentum = .01		# The momentum value to escape local minima
		self.prevDelta = []		# The previous layer's delta during back propogation
		self.currentError = []		# The error from the last epoch
		self.currentOutput = []		# The current output for all training data points. Updated individually for stochastic learning
		self.totalEpochs = 0		# The total number of epochs to train over per session
		self.trainingInput = 0		# The current input values for training (only used in stochastic training)
	
		# Th matrix of weights
		# Each entry is connection between layers
		self.weights = []

		# The matrix of neurons
		# Each entry is a layer of neurons
		self.neurons = []

		# The matrix of biases
		#Each entry is a bias for a layer
		self.biases = []

	# Print information on how to use this program
	# USAGE:
	# python MLP.py <data file> <activation function> <learning type> <learning rate> 
	# <epochs> <input shape> <hidden layer 1 shape> ... <output shape>
	def printUsage(self):
		print "USAGE:"
		print "python MLP.py <data file> <activation function> <learning type> <learning rate> <epochs> <input shape> <hidden layer 1 shape> ... <output shape>\n"
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

	#----------------------------------------#
	######## Initialization Functions ########
	#----------------------------------------#

	# Read the data from the given file
	# Initialize the first layer of neurons
	# Initialize the y matrix with training data answers
	# Print an error if the matrix size doesnt match with the data
	# Print an error if there is an error in the training data
	def initTrainingData(self, trainingData):
		dataSize = (int)(trainingData.readline())	# Get size of training data to use
		temp = []
		temp2 = []
		tempy = []					# Holds the explicit answers to the training data
		distinctLabels = 0

		for dataPoint in range(0, dataSize):
			line = trainingData.readline()		# Get next line of training data
			line = line.rstrip("\n")

			if not line:
				print "The Training Data File Is Incorrectly Formated\n"
				sys.exit(0)

			trainingInfo = line.split(":")		# Split line into both the input data and correct answer
			data = trainingInfo[0].split(",")
			answer = trainingInfo[1]

			if len(data) != (int)(self.data[6]):
				print "The Training Data File Is Incorrectly Formated\n"
				sys.exit(0)

			if answer not in tempy:
				self.labels[distinctLabels] = answer	# Add label to labels with key index of 1 in onehot represntation
				distinctLabels += 1
					
			data = map(float, data)
			temp.append(data)

			if self.learningType == 1 and dataPoint == 0:
				temp2.append(data)

			tempy.append(answer)

		if self.learningType == 0:			 	# Initialize input neurons for batch training
			self.neurons.append(np.array(temp))

		elif self.learningType == 1:				# Initialize input neurons wof stochastic training
			self.neurons.append(np.array(temp2))

		self.x = np.array(temp)					# Initialize array of input training data
		self.yOneHot(distinctLabels, tempy)			# Generate one hot values for answers from explicit answers

	# Initialize one hot values for all training outputs (y)
	# Takes in training outputs with explicit values
	def yOneHot(self, numLabels, y):
		for index in range(0, len(y)):
			answer = np.zeros(numLabels)
			temp = 0
			for label in self.labels.values():
				if label == y[index]:
					answer[temp] = 1
					self.y.append(np.array(answer))
					break
				temp += 1

		self.y = np.array(self.y)
	
	# Read user given argumnts
	# Set global arguments base on given arguments
	# Print error and usage if argument list is invalid
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
	# Do not initialize input neurons, initialized elsewhere
	def initNeurons(self):
		for layer in range(1, self.layers):
			neuronLayer = []
			for neuron in range(0, self.shape[layer]):
				neuronLayer.append(0)

			self.neurons.append(np.array(neuronLayer))
	
	# Initialize Bias list using a uniform distribution
	def initBiases(self):
		for layer in range(0, self.layers - 1):
			bias = np.random.uniform(.01, .1, size = None)
			self.biases = np.append(self.biases, bias)

	# Initialize persistent arrays for storing error and current output
	def initPersistent(self):
		for example in range(0, len(self.x)):
			trainingError = []
			for index in range(0, len(self.y[0])):
				trainingError.append(1.0)

			self.currentError.append(trainingError)
			self.currentOutput.append(trainingError)

		self.currentError = np.array(self.currentError)
		self.currentOutput = np.array(self.currentOutput)

	# Fully initialize Neural Net
	def initNeuralNet(self):
		self.parseArgs()
		self.initWeights()
		self.initNeurons()
		self.initBiases()
		self.initPersistent()
		#print "\nweights\n" + str(self.weights)
		#print "\nbiases\n" + str(self.biases)
		#print "\nneurons\n" + str(self.neurons)
		#print "\nx\n" + str(self.x)
		#print "\ny\n" + str(self.y)

	#----------------------------------------------------#
	################ Activation Functions ################
	#----------------------------------------------------#

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

	# ReLU Activation Function
	def ReLU(self, x):
		return np.maximum(0, x)

	# Derivative of ReKU Function
	def ReLUDerivative(self, x):
		return np.greater(x, 0).astype(int)

	# Leaky ReLU Activation Function
	def LReLU(self, x):
		x[x < 0] *= .01
		return x

	# Derivative of Leaky ReLU functoon
	def LReLUDerivative(self, x):
		x[x < 0] *= .01
		x[x > 0] = (int)(1)
		return x

	# Softmax Activation Function
	def softmax(self, x):	
		norm = x - np.max(x)
		numerator = np.exp(norm) / np.sum(np.exp(norm))
		if numerator.ndim > 1:
			denominator = np.reshape(np.sum(numerator, axis=1), (-1, 1))
			return numerator / denominator
		return numerator

	# Cross Entropy Log Loss Function
	def crossEntropy(self, x):
		m = self.y.shape[0]
		loss = -np.sum(self.y * np.log(x + 1e-12)) / m
		return loss

	#----------------------------------------#
	########### Training Functions ###########
	#----------------------------------------#

	# Check to see if the desired accuracy has been achieved
	# To be used for simple linear classification
	def checkOutput(self):
		if np.mean(np.abs(self.currentError)) < self.desiredAccuracy:
			return 1

		return 0

	# Feed Forward Algorithm
	# Calculates the value(s) of the next layer
	def forwardPass(self, inputLayer, inputWeights, layerBias, classify):
		nextLayer = np.dot(inputLayer, inputWeights)
		nextLayer += layerBias
		if classify:
			nextLayer = self.softmax(nextLayer)

			if self.learningType == 0:
				self.currentOutput = nextLayer

			elif self.learningType == 1:
				self.currentOutput[self.trainingInput] = nextLayer
		
		elif self.activationFunction == 0:
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
				self.currentError = error

			elif self.learningType == 1:
				error = self.y[self.trainingInput] - layer
				self.currentError[self.trainingInput] = error

			delta = error

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
				self.neurons[layer + 1] = self.forwardPass(self.neurons[layer], self.weights[layer], self.biases[layer], layer == self.layers - 2) 

			#Check if we reached desired accuracy
			if epochElapsed == self.maxEpoch:
				break

			#Inccrease epoch elapsed
			epochElapsed += 1

			#Print the error every 10000 epochs
			if epochElapsed % 10000 == 0 or epochElapsed == 1:
				print "Current Session Epoch: " + str(self.totalEpochs + epochElapsed) + " | Error:" + str(np.mean(np.abs(self.currentError)))	

			if epochElapsed % 100 == 0:
				print "Loss: " + str(self.crossEntropy(self.currentOutput))

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
				self.trainingInput = np.random.randint(0, len(self.x), None)
				self.neurons[0][0] = self.x[self.trainingInput]

			if self.checkOutput():
				break

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

				self.neurons[1] = self.forwardPass(inputs, self.weights[0], self.biases[0], False)
				for layer in range(1, self.layers - 1):
					self.neurons[layer + 1] = self.forwardPass(self.neurons[layer], self.weights[layer], self.biases[layer], layer == self.layers - 2)

				maxProb = -1
				hotIndex = -1
				for hot in range(0, len(self.neurons[self.layers - 1])):
					if self.neurons[self.layers - 1][hot] > maxProb:
						maxProb = self.neurons[self.layers - 1][hot]
						hotIndex = hot

				print "\nPREDICTION: " + self.labels.get(hotIndex) + " | CONFIDENCE: " + str(maxProb)

			elif userCommand == "train":
				print "Previous desired accuracy was " + str(self.desiredAccuracy)
				newAccuracy = input("Please enter a new desired accuracy value.\n")
				self.desiredAccuracy = newAccuracy
				requestedEpochs = input("Please enter the maximum epochs to train over.\n")
				self.train(requestedEpochs)

			elif userCommand == "print":
				self.printNet()

			elif userCommand == "error":
				print "Error:" + str(np.mean(np.abs(self.currentError)))

			else:
				print "That is not a valid command.\n"

net = Net()
net.initNeuralNet()
net.train(None)
net.run()
