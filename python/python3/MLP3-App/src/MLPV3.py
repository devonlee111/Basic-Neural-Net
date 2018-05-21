import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import threading
import sys

class Net:

	def __init__(self):
		self.args = []
		self.init = False

		self.lock = threading.Lock()

		# The activation function to use for each layer
		# 0 = tanh function
		# 1 = sigmoid function
		# 2 = ReLU function
		# 3 = Leaky ReLU function
		self.activationFunction = 0

		# How we should train the neural net on the data
		# 0 = batch learning
		# 1 = stochastic learning
		# 2 = mini batch learning
		self.learningType = 0

		self.x = []			# Matrix of all the training data inputs
		self.y = []			# one hot matrix of answers to the training data
		self.labels = dict()		# Dictionary linking index of one hot value to answer
		self.layers = 0			# The number of layers in the neural net
		self.shape = []			# The number of neurons in each layer
		self.lr = .001			# The learning rate of the neural net
		self.desiredAccuracy = 0.5	# The threshold accuracy to stop training
		self.momentum = .01		# The momentum value to escape local minima
		self.prevDelta = []		# The previous layer's delta during back propogation
		self.currentError = []		# The current error for all training data points. Updated individually for stochastic learning
		self.currentOutput = []		# The current output for all training data points. Updated individually for stochastic learning
		self.loss = 0.0
		self.epochElapsed = 0		# The total number of epochs trained over
		self.epochs = 0			# The max number of epochs to train over

		self.trainingInput = 0		# The current input values for training (only used in stochastic training)
		self.batchSize = 100		# The size of the mini batch to use in training

		# The matrix of weights
		# Each entry is a connection between layers
		self.weights = []

		# The matrix of neurons
		# Each Entry is a layer of neurons
		self.neurons = []

		# The matrix of biases
		# Each entry is a bias for a layer
		self.biases = []

		self.run = False

	#----------------------------------------#
	######## Initialization Functions ########
	#----------------------------------------#

	# Check if the neural net has been initialized yet
	def isInit(self):
		return self.init

	# Set neural net initilized variable
	def setInit(self, init):
		self.init = init

	# Completely wipe information from the neural net
	def clearNet(self):
		self.activationFunction = 0
		self.learningType = 0
		self.x = []
		self.y = []
		self.labels = dict()
		self.layers = 0
		self.shape = []
		self.lr = .001
		self.desiredAccuracy = 0.5
		self.momentum = .01
		self.prevDelta = []
		self.currentError = []
		self.currentOutput = []
		self.loss = 0.0
		self.epochElapsed = 0
		self.epochs = 0
		self.trainingInput = 0
		self.weights = []
		self.neurons = []
		self.biases = []
		self.init = False
		self.batchSize = 100
		self.run = False

	# Returns the shape of the neural net
	def getShape(self):
		return self.shape

	# Returns the neural net's weight matrix
	def getWeights():
		return self.weights

	# Read the data from the given file
	# Initialize the first layer of neurons
	# Initialize the y matrix with training data answers
	# Print an error if the matrix size doesnt match with the data
	# Print an error if there is an error in the training data
	def initTrainingData(self, trainingData):
		dataSize = (int)(trainingData.readline())
		temp = []
		temp2 = []
		tempy = []
		distinctLabels = 0

		for dataPoint in range(0, dataSize):
			self.randomOrder.append(dataPoint)
			line = trainingData.readline()
			line = line.rstrip("\n")

			if not line:
				print("The Training Data File Is Incorrectly Formated\n")
				sys.exit(0)
	
			trainingInfo = line.split(":")	
			data = trainingInfo[0].split(",")
			answer = trainingInfo[1]

			if len(data) != (int)(self.shape[0]):
				print("The Training Data File Is Incorrectly Formated\n")
				sys.exit(0)

			if answer not in tempy:
				self.labels[distinctLabels] = answer
				distinctLabels += 1

			data = list(map(float, data))
			temp.append(data)

			if (self.learningType == 1 and dataPoint == 0) or (self.learningType == 2 and dataPoint < self.batchSize):
				temp2.append(data)
			
			tempy.append(answer)

		if self.learningType == 0:
			self.neurons.append(np.array(temp))

		else:
			self.neurons.append(np.array(temp2))

		self.x = np.array(temp)
		self.yOneHot(distinctLabels, tempy)

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

	# Initialize weight matrix using a uniform distribution
	def initWeights(self):
		for layer in range(0, self.layers - 1):
			layerWeights = np.random.uniform(.01, .1, size = (self.shape[layer], self.shape[layer + 1]))
			self.weights.append(layerWeights)

	# Initialize neuron matrix to all 0
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

	def initPersistent(self):
		for example in range(0, len(self.x)):
			trainingError = []
			for index in range(0, len(self.y[0])):
				trainingError.append(1.0)

			self.currentError.append(trainingError)
			self.currentOutput.append(trainingError)

		self.currentError = np.array(self.currentError)
		self.currentOutput = np.array(self.currentOutput)

	def setLearningRate(self, lr):
		self.lr = (float)(lr)

	def setEpochs(self, epochs):
		self.epochs = (int)(epochs)
		self.epochElapsed = 0

	def setMaxEpochs(self, epochs):
		self.epochs = (int)(epochs)

	def getEpochs(self):
		return self.epochElapsed

	def shouldContinue(self):
		if self.epochElapsed >= self.epochs or self.checkOutput() == 1 or self.run == False:
			return False

		return True

	def setError(self, error):
		self.error = (float)(error)

	def getError(self):
		error = self.currentError
		return np.mean(np.abs(error))

	def getLoss(self):
		return self.loss

	def setActivationFunction(self, activationFunction):
		if activationFunction == "Tanh":
			self.activationFunction = 0
			
		elif activationFunction == "Sigmoid":
			self.activationFunction = 1
			
		elif activationFunction == "ReLU":
			self.activationFunction = 2
			
		elif activationFunction == "LReLU":
			self.activationFunction = 3

	def setLearningType(self, learningType):
		if learningType == "Batch":
			self.learningType = 0

		elif learningType == "Stochastic":
			self.learningType = 1

		elif learningType == "Mini Batches":
			self.learningType = 2

	def setShape(self, shape):
		self.shape = shape.split()
		self.shape =list(map(int, self.shape))
		self.layers = len(self.shape)

	def setBatchSize(self, size):
		self.batchSize = int(size)

	def getBatchSize(self):
		return self.batchSize	

	# Fully Initialized Neural Net
	def initNeuralNet(self, lr, epochs, error, activationFunction, learningType, trainingData, shape, batchSize):
		self.clearNet()
		self.setLearningRate(lr)
		self.setEpochs(epochs)
		self.setError(error)
		self.setActivationFunction(activationFunction)
		self.setLearningType(learningType)
		self.setBatchSize(batchSize)
		self.setShape(shape)
		self.initTrainingData(open(trainingData))
		self.initWeights()
		self.initNeurons()
		self.initBiases()
		self.initPersistent()

	# Change the configuration of the Neural Net
	def editNet(self, lr, epochs, error):
		self.setLearningRate(lr)
		self.setMaxEpochs(epochs)
		self.setError(error)

	def startNet(self):
		self.run = True

	def stopNet(self):
		self.run = False

	def isRunning(self):
		return self.run

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
		numerator = np.exp(norm) / np.sum(np.exp(norm))
		if numerator.ndim > 1:
			denominator = np.reshape(np.sum(numerator, axis=1), (-1, 1))
			return numerator / denominator

		return np.exp(x) / float(sum(np.exp(x)))

	def crossEntropy(self, x):
		m = self.y.shape[0]
		loss = -np.sum(self.y * np.log(x + 1e-12)) / m
		return loss

	def generateGraph(self, title):
		errorPlot, = plt.plot(self.epochHistory, self.errorHistory, marker = 'o', label = 'Error')
		lossPlot, = plt.plot(self.epochHistory, self.lossHistory, marker = '^', label = 'Loss')
		plt.legend([errorPlot, lossPlot],['Error', 'Loss'])
		plt.suptitle(title)
		plt.xlabel('Epoch')
		plt.ylabel('Value')
		plt.savefig('plot.png')	

	#Check to see if the desired accuracy has been achieved
	def checkOutput(self):
		if np.mean(np.abs(self.currentError)) < self.error:
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

			elif self.learningType == 2:
				for index in range(0, self.batchSize):
					self.currentOutput[self.randomOrder[index]] = nextLayer[index]

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
		error = -1
		if layerNum == self.layers - 1:
			if self.learningType == 0:
				error = self.y - layer
				self.currentError = error

			elif self.learningType == 1:
				error = self.y[self.trainingInput] - layer
				self.currentError[self.trainingInput] = error

			elif self.learningType == 2:
				error = self.y[self.trainingInput : self.trainingInput + self.batchSize] - layer
				self.currentError[self.trainingInput : self.trainingInput + self.batchSize] = error

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

		if not self.prevDelta or self.learningType == 0 or self.learningType == 2:
			self.weights[layerNum - 1] += prevLayer.T.dot(delta) * lr

		else:
			self.weights[layerNum - 1] += (prevLayer.T.dot(delta) * lr) + (self.prevDelta[self.layers - layerNum - 1] * self.momentum)

		self.biases[layerNum - 1] += np.sum(delta) * lr
		return delta

	# Train the neural net on a data set
	def trainingPass(self):
		#Forward Propogation
		for layer in range(0, self.layers - 1):
			print(len(self.neurons[0]))
			self.neurons[layer + 1] = self.forwardPass(self.neurons[layer], self.weights[layer], self.biases[layer], layer == self.layers - 2) 

		if self.checkOutput() == 1:
			return -1

		self.loss = self.crossEntropy(self.currentOutput)

		#Inccrease epoch elapsed
		self.epochElapsed += 1
		if self.epochElapsed % 100 == 0:
		 	loss = self.crossEntropy(self.currentOutput)
			error = np.mean(np.abs(self.currentError))
			self.lossHistory.append(loss)
			self.errorHistory.append(error)
			self.epochHistory.append(self.epochElapsed)

		#Back Propagation
		delta = None
		temp = []
		#self.prevDelta = None

		for layer in range(self.layers - 1, 0, -1):
			inputWeights = None
			if layer != (self.layers - 1):
				inputWeights = self.weights[layer]

			delta = self.backwardPass(layer, self.neurons[layer], self.neurons[layer - 1], inputWeights, delta, self.lr)
			temp.append(delta)

		if self.learningType == 1:
			self.trainingInput = np.random.randint(0, len(self.x), None)
			self.neurons[0][0] = self.x[self.trainingInput]

		elif self.learningType == 2:
			np.random.shuffle(self.randomOrder)
			for index in range(0, self.batchSize):
				self.neurons[0][index] = self.x[self.randomOrder[index]]

		self.prevDelta = temp

	def train(self):
		while(self.shouldContinue()):
			self.trainingPass()
