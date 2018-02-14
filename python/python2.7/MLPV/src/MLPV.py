import numpy as np
import sys

class Net:

	def __init__(self):
		self.args = []
		self.y = []
		self.x = []
		self.weights = []
		self.neurons = []
		self.biases = []
		self.init = False

	##########Functions	

	def isInit(self):
		return self.init

	def getShape(self):
		return self.shape

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

	def setLearningRate(self, lr):
		self.lr = (float)(lr)

	def setEpochs(self, epochs):
		self.epochs = (int)(epochs)
		self.totalEpochs = 0

	def setError(self, error):
		self.error = (float)(error)

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

	def setShape(self, shape):
		self.shape = shape.split()
		self.shape = map(int, self.shape)
		self.layers = len(self.shape)

	# Read the data from the given file
	# Initialize the first layer of neurons
	# Initialize the y matrix with training data answers
	# Print an error if the matrix size doesnt match with the data
	# Print an error if there is an error in the training data
	def initTrainingData(self, trainingDataPath):
		trainingData = open(trainingDataPath, 'r')
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

			if len(data) != (int)(self.shape[0]) or (len(answer) != (int)(self.shape[self.layers - 1])):
				print "The Training Data File Is Incorrectly Formated\n"
				sys.exit(0)
	
			data = map(float, data)
			temp.append(data)
			if self.learningType == 1 and dataPoint == 0:
				temp2.append(data)

			answer = map(float, answer)
			self.y.append(answer)

		if self.learningType == 0:
			self.neurons.append(np.array(temp))

		elif self.learningType == 1:
			self.neurons.append(np.array(temp2))

		self.x.append(np.array(temp))
		self.trainingInput = 0;

	#Fully Initialized Neural Net
	def initNeuralNet(self, lr, epochs, error, activationFunction, learningType, trainingData, shape):
		self.setLearningRate(lr);
		self.setEpochs(epochs)
		self.setError(error)
		self.setActivationFunction(activationFunction)
		self.setLearningType(learningType)
		self.setShape(shape);
		self.initTrainingData(trainingData)
		self.initWeights()
		self.initNeurons()
		self.initBiases()
		self.isInit = True

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
		return np.exp(x) / float(sum(np.exp(x)))

	#Check to see if the desired accuracy has been achieved
	def checkOutput(self):
		if np.mean(np.abs(self.currentError)) < self.error:
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
		error = -1
		if layerNum == self.layers - 1:
			if self.learningType == 0:
				error = self.y - layer

			elif self.learningType == 1:
				error = self.y[self.trainingInput] - layer

			self.currentError = error

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
		if not self.prevDelta or learningType == 0:
			self.weights[layerNum - 1] += prevLayer.T.dot(delta) * lr

		else:
			self.weights[layerNum - 1] += (prevLayer.T.dot(delta) * lr) + (prevDelta[self.layers - layerNum - 1] * momentum)

		self.biases[layerNum - 1] += np.sum(delta) * lr
		return delta

	# Train the neural net on a data set
	def train(self):
		self.currentError = -1
		consecErrorReached = 0
		epochElapsed = 0

		while True:
			#Forward Propogation
			for layer in range(0, self.layers - 1):
				self.neurons[layer + 1] = self.forwardPass(self.neurons[layer], self.weights[layer], self.biases[layer]) 

			#Check if we reached desired accuracy
			if epochElapsed == self.epochs:
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
			self.prevDelta = None
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
		print self.neurons[self.layers - 1]
