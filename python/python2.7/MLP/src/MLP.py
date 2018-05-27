import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
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
		# 2 = minibatches
		self.learningType = 0

		self.x = []			# Matrix of all training data inputs
		self.y = []			# one hot matrix of answers to training data
		self.testInputs = []		# Matrix of all testing data inputs
		self.testHot = []		# one hot matrix of answers to testing input
		self.labels = dict()		# Dictionary linking index of one hot value to answer
		self.layers = 0			# The number of layers in the neural net
		self.shape = []			# The number of neurons in each layer
		self.lr = 0.001			# The learning rate of the neural network
		self.desiredAccuracy = .03	# The threshhold accuracy to stop training
		self.momentum = .01		# The momentum value to escape local minima
		self.prevDelta = []		# The previous layer's delta during back propogation
		self.currentError = []		# The error from the last epoch
		self.currentOutput = []		# The current output for all training data points. Updated individually for stochastic learning
		self.totalEpochs = 0		# The total number of epochs to train over per session
		self.trainingInput = 0		# The current input values for training (only used in stochastic training)
		self.batchSize = 100		# The size of the batches to use when training using mini batches
		self.randomOrder = []		# List used to choose random data points to train with when using mini batches
		self.lossHistory = []		# Keeps track of the loss history for graphing purposes
		self.errorHistory = []		# Keeps track of the error history for graphing purposes
		self.epochHistory = []		# Keeps track of the epoch history for graphing purposes
		self.initialEpoch = 50000	# The number of epochs to train over
		self.trainingData = None
		self.testingData = None


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
		print "python MLP.py <data file> <arg 1> <arg 2> ..."
		print "\"data file\" and \"layers\" argument are required."
		print "Use -help or -h for more information on arguments."

	# Show the help menu to the user
	# Gives information on different program arguments
	def printHelp(self):
		while(True):
			print "HELP MENU"
			print "1) activation functions"
			print "2) training type"
			print "3) training parameters"
			print "4) neural net layer shapes"
			print "5) test file"
			print "6) help"
			print "q - quit help"
			selection = raw_input()
			
			if selection == "q" or selection == "quit":
				sys.exit(0)

			selection = int(selection)

			if selection == 1:
				print "\nACTIVATION FUNCTIONS"
				print "tanh"
				print "sigmoid"
				print "relu"
				print "lrelu"

			elif selection == 2:
				print "\nTraining Type"
				print "batch"
				print "minibatch"
				print "stochastic"

			elif selection == 3:
				print "\nTraining Parameters"
				print "lr=<value>"
				print "epochs=<value>"
				print "batchsize=<value>"
				print "testfile=<path>"

			elif selection == 4:
				print "\nLayer Shapes"
				print "layers=<hidden layer 1 size>,<hidden layer 2 size>,<hidden layer 3 size>..."

			elif selection == 5:
				print "\nTest File"
				print "testfile=<path>\tUsed to get a more accurate error and loss by more accurately tracking learning progress"

			elif selection == 6:
				print "\nMore Help"
				print "input index of desired help menu item to get a list of related arguments"

			print "press any key to continue"
			raw_input()

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
	def initTrainingData(self):
		dataSize = (int)(self.trainingData.readline())	# Get size of training data to use
		temp = []
		temp2 = []
		tempy = []					# Holds the explicit answers to the training data
		distinctLabels = 0

		for dataPoint in range(0, dataSize):
			self.randomOrder.append(dataPoint)
			line = self.trainingData.readline()		# Get next line of training data
			line = line.rstrip("\n")

			if not line:
				print "The Training Data File Is Incorrectly Formated\n"
				sys.exit(0)

			trainingInfo = line.split(":")		# Split line into both the input data and correct answer
			data = trainingInfo[0].split(",")
			answer = trainingInfo[1]

			if self.shape[0] == 0:
				self.shape[0] = len(data)

			elif len(data) != self.shape[0]:
				print "The Training Data Have Differing Sizes\n"
				sys.exit(0)

			if answer not in tempy:
				self.labels[distinctLabels] = answer	# Add label to labels with key index of 1 in onehot represntation
				distinctLabels += 1
					
			data = map(float, data)
			temp.append(data)

			if (self.learningType == 1 and dataPoint == 0) or (self.learningType == 2 and dataPoint < self.batchSize):
				temp2.append(data)

			tempy.append(answer)

		if self.learningType == 0:			 	# Initialize input neurons for batch training
			self.neurons.append(np.array(temp))

		else:							# Initialize input neurons wof stochastic training
			self.neurons.append(np.array(temp2))

		self.shape.append(distinctLabels)
		self.x = np.array(temp)					# Initialize array of input training data
		self.y = self.oneHot(distinctLabels, tempy)			# Generate one hot values for answers from explicit answers

	def initTestingData(self):
		dataSize = (int)(self.testingData.readline())
		temp = []
		answers = []

		for dataPoint in range(0, dataSize):
			line = self.testingData.readline()
			line = line.rstrip("\n")

			if not line:
				print "The Testing Data File Is Incorrectly Formatted\n"
				sys.exit(0)

			testingInfo = line.split(":")
			data = testingInfo[0].split(",")
			answer = testingInfo[1]

			data = map(float, data)
			temp.append(data)
			answers.append(answer)

		self.testInputs = np.array(temp)
		self.testHot = self.oneHot(len(self.labels), answers)

	# Initialize one hot values for all training outputs (y)
	# Takes in training outputs with explicit values
	def oneHot(self, numLabels, y):
		onehot = []
		for index in range(0, len(y)):
			answer = np.zeros(numLabels)
			temp = 0
			for label in self.labels.values():
				if label == y[index]:
					answer[temp] = 1
					onehot.append(np.array(answer))
					break
				temp += 1

		return np.array(onehot)
	
	# Read user given argumnts
	# Set global arguments base on given arguments
	# Print error and usage if argument list is invalid
	def parseArgs(self):
		if len(sys.argv) < 2:
			print "not enough args."
			self.printUsage()
			sys.exit(0)

		if "-help" in self.data or "-h" in self.data:
			self.printHelp()
			sys.exit(0)

		try:
			self.trainingData = open(self.data[1], 'r')

		except IOError:
			print "The Given Training Data File, " + str(self.data[1]) + ", Could Not Be Found Or Opened\n"
			self.printUsage()
			sys.exit(0)

		numActivationFunction = 0
		numLearningType = 0
		numLearningRate = 0
		numGivenEpochs = 0
		numBatchSize = 0

		for index in range(2, len(self.data)):
			arg = self.data[index]
			
			if arg == "tanh":
				self.activationFunction = 0
				numActivationFunction += 1

			elif arg == "sigmoid":
				self.activationFunction = 1
				numActivationFunction += 1

			elif arg == "relu" or self.data[2] == "ReLU":
				self.activationFunction = 2
				numActivationFunction += 1

			elif arg == "lrelu" or self.data[2] == "LReLU":
				self.activationFunction = 3
				numActivationFunction += 1

			elif arg == "batch":
				self.learningType = 0
				numLearningType += 1

			elif arg == "stochastic":
				self.learningType = 1
				numLearningType += 1

			elif arg == "mini batch" or arg == "minibatch" or arg == "mbatch" or arg == "minib":
				self.learningType = 2
				numLearningType += 1

			elif "batchsize=" in arg:
				self.batchSize = int(arg.split("=")[1])
				numBatchSize += 1

			elif "lr=" in arg:
				self.lr = float(arg.split("=")[1])
				numLearningRate += 1

			elif "epochs=" in arg:
				self.initialEpoch = int(arg.split("=")[1])
				numGivenEpochs += 1

			elif "layers=" in arg:
				layerSizes = arg.split("=")[1]
				layerSizes = layerSizes.split(",")
				self.layers = len(layerSizes) + 2
				self.shape.append(0)

				for layer in range(0, len(layerSizes)):
					self.shape.append((int)(layerSizes[layer]))

			elif "testfile=" in arg:
				try:
					self.testingData = open(self.data[1], 'r')

				except IOError:
					print "The Given Training Data File, " + str(self.data[1]) + ", Could Not Be Found Or Opened\n"
					self.printUsage()
					sys.exit(0)

		if numActivationFunction > 1:
			print "Only One Activation Function May Be Specified!"
			print "Found " + str(numActivationFunction) + " Activation Functions."
			sys.exit(0)
		
		if numLearningType > 1:
			print "Only One Learning Type May Be Specified!"
			print "Found " + str(numLearningType) + " Learning Types."
			sys.exit(0)

		if numLearningRate > 1:
			print "Only One Learning Rate May Be Specified!"
			print "Found " + str(numLearningRate) + " Learning Rate."
			sys.exit(0)

		if numGivenEpochs > 1:
			print "Only One Epoch Value May Be Specified!"
			print "Found " + str(numGivenEpochs) + " Epoch Values."
			sys.exit(0)

		if numBatchSize > 1:
			print "Only One Batch Size Value May Be Specified!"
			print "Found " + str(numBatchSize) + " Batch Sizes."
			sys.exit(0)

		if len(self.shape) == 0:
			print "Neural Net Shape Must Be Specified!"
			sys.exit(0)

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
		#print "Start Init"
		self.parseArgs()
		self.initTrainingData()

		if self.testingData is not None:
			self.initTestingData()

		self.initWeights()
		self.initNeurons()
		self.initBiases()
		self.initPersistent()
		#print "End Init"
		#print "\nweights\n" + str(self.weights)
		#print "\nbiases\n" + str(self.biases)
		#print "\nneurons\n" + str(self.neurons)
		#print "\nx\n" + str(self.x)
		#print "\ny\n" + str(self.y)
		#print "\ncurrent output\n" + str(self.currentOutput)
		#print "\ncurrentError\n" + str(self.currentError)

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
	def crossEntropy(self, x, y):
		m = y.shape[0]
		loss = -np.sum(y * np.log(x + 1e-12)) / m
		return loss

	#----------------------------------------#
	########### Training Functions ###########
	#----------------------------------------#

	# Check to see if the desired accuracy has been achieved
	# To be used for simple linear classification
	def checkOutput(self, error):
		if np.mean(np.abs(error)) < self.desiredAccuracy:
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
		if layerNum == self.layers - 1:
			if self.learningType == 0:
				error = self.y - layer
				self.currentError = error

			elif self.learningType == 1:
				error = self.y[self.trainingInput] - layer
				self.currentError[self.trainingInput] = error

			elif self.learningType == 2:
				error = []
				for index in range(0, self.batchSize):
					error.append(self.y[self.randomOrder[index]] - layer[index])
					self.currentError[self.randomOrder[index]] = error[index]

				error = np.array(error)

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
			self.weights[layerNum - 1] += prevLayer.T.dot(delta) * self.lr

		else:
			self.weights[layerNum - 1] += (prevLayer.T.dot(delta) * self.lr) + (self.prevDelta[self.layers - layerNum - 1] * self.momentum)

		self.biases[layerNum - 1] += np.sum(delta) * lr
		return delta

	# Train the neural net on a data set
	def train(self, requestedEpoch):
		epochElapsed = 0

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

			elif self.learningType == 2:
				np.random.shuffle(self.randomOrder)
				for index in range(0, self.batchSize):
					 self.neurons[0][index] = self.x[self.randomOrder[index]]

			self.prevDelta = temp

			#Inccrease epoch elapsed
			epochElapsed += 1

			if epochElapsed % 100 == 0:
				if self.testingData != None:
					error = self.testingPass(epochElapsed)

				else :
					loss = self.crossEntropy(self.currentOutput, self.y)
					error = np.mean(np.abs(self.currentError))
					self.printTrainingStatus(loss, error, epochElapsed)

				if self.checkOutput(error):
					break

		self.totalEpochs += epochElapsed
		print "\nNumber of epochs in current training session: ",
		print epochElapsed
		print "Total epochs over all training sessions: ",
		print self.totalEpochs

	def testingPass(self, epochElapsed):
		self.neurons[1] = self.forwardPass(self.testInputs, self.weights[0], self.biases[0], False)
		
		for layer in range(1, self.layers - 1):
			self.neurons[layer + 1] = self.forwardPass(self.neurons[layer], self.weights[layer], self.biases[layer], layer == self.layers - 2)

		error = self.testHot - self.neurons[self.layers - 1]
		error = np.mean(np.abs(error))
		loss = self.crossEntropy(self.currentOutput, self.y)
		self.printTrainingStatus(loss, error, epochElapsed)
		return error

	def printTrainingStatus(self, loss, error, epochElapsed):
		self.lossHistory.append(loss)
		self.errorHistory.append(error)
		self.epochHistory.append(epochElapsed)
		sys.stdout.write("Epoch: %d \t| Loss: %.4f | Error: %.4f\r" % (self.totalEpochs + epochElapsed, loss, error))
		sys.stdout.flush()

	def run(self):
		while True:
			userCommand = raw_input("\nNeural Net has finished its training. Here is a list of available commands.\n\"predict\"\tNeural Net is ready to process and predict your input.\n\"train\"\t\tNeural Net may be trained further.\n\"print\"\t\tPrint the values of the neural net.\n\"error\"\t\tPrint the current error.\n\"graph\"\t\tCreate a loss and error graph png\n\"quit\"\t\tQuit the program\n--> ")
			if userCommand == "quit" or userCommand == "exit" or userCommand == "q":
				print "Thank you for using my neural net program.\n"
				sys.exit(0)

			elif userCommand == "predict":
				inputs = []
				isFile = False
				for netInput in range(0, len(self.neurons[0][0])):
					newInput = raw_input("Please enter the next input datum.\n")
					try:
						f = open(newInput, "r")
						isFile = True
						break;

					except:
						inputs.append(int(newInput))

				if isFile:
					temp = []
					line = f.readline()
					line = line.rstrip("\n")
					data = line.split(",")
					data = map(float, data)
					inputs = np.array(data)

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
				if self.testingData != None:
					self.neurons[1] = self.forwardPass(self.testInputs, self.weights[0], self.biases[0], False)

					for layer in range(1, self.layers - 1):
						self.neurons[layer + 1] = self.forwardPass(self.neurons[layer], self.weights[layer], self.biases[layer], layer == self.layers - 2)

					error = self.testHot - self.neurons[self.layers - 1]
					error = np.mean(np.abs(error))
	
				else:
					error = np.mean(np.abs(self.currentError))

				print "Error:" + str(error)

			elif userCommand == "graph":	
				errorPlot, = plt.plot(self.epochHistory, self.errorHistory, marker = 'o', label = 'Error')
				lossPlot, = plt.plot(self.epochHistory, self.lossHistory, marker = '^', label = 'Loss')
				plt.legend([errorPlot, lossPlot],['Error', 'Loss'])
				plt.suptitle(self.data[1])
				plt.xlabel('Epoch')
				plt.ylabel('Value')
				plt.savefig('plot.png')			

			else:
				print "That is not a valid command.\n"

net = Net()
net.initNeuralNet()
net.train(None)
net.run()
