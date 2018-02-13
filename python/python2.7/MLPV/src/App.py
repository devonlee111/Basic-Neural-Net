import Tkinter as tk
import threading
import MLPV

class App(threading.Thread):

	lr = .01
	learningType = 0
	maxEpochs = 10000
	restart = False

	def __init__(self):
		self.args = []

		# Create the root window.
		self.root = tk.Tk()
		self.root.title("Neural Net Visualization")
		self.root.geometry("%dx%d+0+0" % (self.root.winfo_screenwidth(), self.root.winfo_screenheight()))

		# Create the frame for the canvas.
		netFrame = tk.Frame(self.root, highlightbackground="black", highlightcolor="black", highlightthickness=1)
		netFrame.pack(side = tk.RIGHT, fill = tk.BOTH, expand = 1)

		# Create the canvas for the net representation.
		net = tk.Canvas(netFrame)
		net.configure(background='pink')
		net.pack(fill = tk.BOTH, expand = 1)

		# Create the frame for the settings.
		settingsFrame = tk.Frame(self.root)
		settingsFrame.pack(side = tk.LEFT, fill = tk.Y)

		# Create a slider to select learning rate.
		self.lrSlider = tk.Scale(settingsFrame, label = "Learning Rate", from_ = 0, to = 1, orient = tk.HORIZONTAL, resolution = 0.001, command = self.setLearningRate)
		self.lrSlider.grid(row = 0, column = 0, sticky = tk.W)

		# Create an entry box for custom learning rate.
		self.lrInput = tk.Entry(settingsFrame)
		self.lrInput.grid(row = 1, column = 0, sticky = tk.W, pady = 10)

		# Create a slider to select number of epochs.
		self.epochSlider = tk.Scale(settingsFrame, label = "Training Epochs", from_ = 1000, to = 1000000, orient = tk.HORIZONTAL, length = 200, resolution = 1000, command = self.setEpochs)
		self.epochSlider.grid(row = 2, column = 0, sticky = tk.W)

		# Create an entry box for a custom epochs.
		self.epochInput = tk.Entry(settingsFrame)
		self.epochInput.grid(row = 3, column = 0, sticky = tk.W, pady = 10)

		# Create an slider to select the error to stop at.
		self.errorSlider = tk.Scale(settingsFrame, label = "Desired Error", from_ = 0, to = .3, orient = tk.HORIZONTAL, length = 200, resolution = 0.01, command = self.setError)
		self.errorSlider.grid(row = 4, column = 0, sticky = tk.W)

		# Create an entry box for a custom error.
		self.errorInput = tk.Entry(settingsFrame)
		self.errorInput.grid(row = 5, column = 0, sticky = tk.W, pady = 10)

		# Create a dropdown menu to select activation function.
		self.activationFunction = tk.StringVar(self.root)
		self.activationFunction.set("Tanh")
		self.activationMenu = tk.OptionMenu(settingsFrame, self.activationFunction, "Tanh", "Sigmoid", "ReLU", "LReLU")
		self.activationMenu.grid(row = 6, column = 0, sticky = tk.W)

		# Create a drowdown menu to select training type.
		self.trainingType = tk.StringVar(self.root)
		self.trainingType.set("Batch")
		self.trainingMenu = tk.OptionMenu(settingsFrame, self.trainingType, "Batch", "Stochastic")
		self.trainingMenu.grid(row = 7, column = 0, sticky = tk.W)	

		self.net = MLPV.Net()

		# Start the window's mainloop.
		self.root.mainloop()

	def getLearningRate(self):
		return self.lr

	def setLearningRate(self, lr):
		self.lrInput.delete(0, tk.END)
		self.lrInput.insert(0, lr)
		self.lr = lr

	def getEpochs():
		return self.epochs

	def setEpochs(self, maxEpochs):
		self.epochInput.delete(0, tk.END)
		self.epochInput.insert(0, maxEpochs)
		self.maxEpochs = maxEpochs
	
	def getError(self):
		return self.error

	def setError(self, error):
		self.errorInput.delete(0, tk.END)
		self.errorInput.insert(0, error)
		self.error = error

	def getActivationFunction(self):
		return self.activationFunction

	def setActivationFunction(self, activationFunction):
		self.activationFunction = activationFunction

	def getLearningType(self):
		return self.learningType

	def setLearningType(self, learningType):
		self.learningType = learningType

app = App()
