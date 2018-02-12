import Tkinter as tk

class Gui:

	lr = .01
	learningType = 0
	maxEpochs = 10000
	restart = False

	def __init__(self):
		self.args = []
		root = tk.Tk()
		root.title("Neural Net Visualization")
		root.geometry("%dx%d+0+0" % (root.winfo_screenwidth(), root.winfo_screenheight()))

		frame = tk.Frame(root)
		frame.pack()
		lrSlider = tk.Scale(root, label = "Learning Rate", from_ = 0, to = 1, orient = tk.HORIZONTAL, length = 200, resolution = 0.001)
		lrSlider.pack()

		epochSlider = tk.Scale(root, label = "Training Epochs", from_ = 1000, to = 1000000, orient = tk.HORIZONTAL, length = 200, resolution = 1000)
		epochSlider.pack()

		errorSlider = tk.Scale(root, label = "Desired Error", from_ = 0, to = .3, orient = tk.HORIZONTAL, length = 200, resolution = 0.01)
		errorSlider.pack()

		activationFunction = tk.StringVar(root)
		activationFunction.set("Tanh")
		activationMenu = tk.OptionMenu(root, activationFunction, "Tanh", "Sigmoid", "ReLU", "LReLU")
		activationMenu.pack()

		trainingType = tk.StringVar(root)
		trainingType.set("Batch")
		trainingMenu = tk.OptionMenu(root, trainingType, "Batch", "Stochastic")
		trainingMenu.pack()


		root.mainloop()

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

