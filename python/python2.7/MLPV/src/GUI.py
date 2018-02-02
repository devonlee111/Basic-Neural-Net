import Tkinter as tk

class Gui:

	lr = .01
	activationFunction = 0
	learningType = 0
	maxEpochs = 10000
	restart = False
	root = tk.Tk()
	frame = tk.Frame(root)
	frame.pack()

	def __init__(self):
		self.args = []

	def getLearningRate():
		return lr

	def setLearningRate(lr):
		self.lr = lr

	def getActivationFunction():
		return activationFunction

	def setActivationFunction(activationFunction):
		self.activationFunction = activationFunction

	def getLearningType():
		return learningType

	def setLearningType(learningType):
		self.learningType = learningType

	def getMaxEpochs():
		return maxEpochs

	def setMaxEpochs(maxEpochs):
		self.maxEpochs = maxEpochs

	root.mainloop();
