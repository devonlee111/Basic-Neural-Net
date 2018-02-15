import Tkinter as tk
import numpy as np
import tkFileDialog
import threading
import MLPV

class App():

	def __init__(self):
		self.args = []

		# Create the root window.
		self.root = tk.Tk()
		self.root.title("Neural Net Visualization")
		self.root.geometry("%dx%d+0+0" % (self.root.winfo_screenwidth(), self.root.winfo_screenheight()))

		# Create the frame for the canvas.
		netFrame = tk.Frame(self.root, highlightbackground = "black", highlightcolor = "black", highlightthickness=1)
		netFrame.pack(side = tk.RIGHT, fill = tk.BOTH, expand = 1)

		# Create the canvas for the net representation.
		self.canvas = tk.Canvas(netFrame)
		self.canvas.configure(background='pink')
		self.canvas.pack(fill = tk.BOTH, expand = 1)

		# Create the frame for the settings.
		settingsFrame = tk.Frame(self.root)
		settingsFrame.pack(side = tk.LEFT, fill = tk.Y)

		# Create a slider to select learning rate.
		self.lrSlider = tk.Scale(settingsFrame, label = "Learning Rate", from_ = 0.001, to = 1, orient = tk.HORIZONTAL, resolution = 0.001, command = self.setLearningRate)
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
		self.learningType = tk.StringVar(self.root)
		self.learningType.set("Batch")
		self.trainingMenu = tk.OptionMenu(settingsFrame, self.learningType, "Batch", "Stochastic")
		self.trainingMenu.grid(row = 7, column = 0, sticky = tk.W)	
		
		# Create a table and entry to input neural net shape
		tk.Label(settingsFrame, text = "Neural Net Shape").grid(row = 8, column = 0, sticky = tk.W);
		self.shapeInput = tk.Entry(settingsFrame)
		self.shapeInput.grid(row = 9, column = 0, sticky = tk.W)

		# Create a button that allows you to select a file for training data.
		self.selectFileBtn = tk.Button(settingsFrame, text = "Select Training Data", command = self.selectTrainingData)
		self.selectFileBtn.grid(row = 10, column = 0, sticky = tk.W, pady = 10)

		self.dataFile = tk.StringVar(self.root)	
		self.dataFile.set("No Data File Selected");
		self.selectedData = tk.Label(settingsFrame, textvariable = self.dataFile, wraplength = 200, anchor = tk.W, justify = tk.LEFT)
		self.selectedData.grid(row = 11, column = 0, sticky = tk.W);

		# Create a button to start network training.
		self.runBtn = tk.Button(settingsFrame, text = "Train", command = self.startNet)
		self.runBtn.grid(row = 12, column = 0, sticky = tk.W)

		self.errorText = self.canvas.create_text(1000, 50, fill = "black", text = "NET NOT STARTED")
		self.epochText = self.canvas.create_text(50, 50, fill = "black", text = "0")

		self.net = MLPV.Net()

	def setLearningRate(self, lr):
		self.lrInput.delete(0, tk.END)
		self.lrInput.insert(0, lr)

	def setEpochs(self, epochs):
		self.epochInput.delete(0, tk.END)
		self.epochInput.insert(0, epochs)
	
	def setError(self, error):
		self.errorInput.delete(0, tk.END)
		self.errorInput.insert(0, error)

	def selectTrainingData(self):
		self.trainingData = tkFileDialog.askopenfilename(initialdir = "/", title = "Select a file", filetypes = (("text files", "*.txt"),))
		self.dataFile.set(self.trainingData)

	def setInputs(self):
		self.lr = self.lrInput.get()
		self.epochs = self.epochInput.get()
		self.error = self.errorInput.get()
		self.shape = self.shapeInput.get()

	def startNet(self):
		self.setInputs()

		if any(c.isalpha() for c in self.lr) or any(c.isalpha() for c in self.epochs) or any(c.isalpha() for c in self.error) or any(c.isalpha() for c in self.shape):
			print "Should not cointain alpha characters."
		else :
			self.net.initNeuralNet(self.lr, self.epochs, self.error, self.activationFunction.get(), self.learningType.get(), self.trainingData, self.shape)


	def drawNet(self):
		shape = self.net.getShape()
		layerDist = self.canvas.winfo_width() / ((len(shape)) + 1)
		layers = []
		
		for layer in range(len(shape)):
			prevNodes = []
			layerPos = (layer + 1) * layerDist
			nodeDist = self.canvas.winfo_height() / (shape[layer] + 1)
			nodes = []
			
			for node in range(shape[layer]):
				nodePos = (node + 1) * nodeDist
				nodes.append(nodePos)
				self.canvas.create_oval(layerPos - 20, nodePos - 20, layerPos + 20, nodePos + 20, fill="black")

				if layer > 0:
					for prevNode in range(shape[layer - 1]):
						line = self.canvas.create_line(layer * layerDist, layers[layer - 1][prevNode], layerPos, nodePos, fill = "blue", width = 5)
						self.canvas.tag_lower(line)	

			layers.append(nodes)

	def updateNetInfo(self):
		error = self.net.getError()
		temp = "Error: \n" + str (error)
		self.canvas.itemconfigure(self.errorText, text = temp)
		epochs = self.net.getEpochs()
		temp = "Epochs: \n" + str (epochs)
		self.canvas.itemconfigure(self.epochText, text = temp)		

	def update(self):
		if self.net.isInit():
			if self.net.shouldContinue():
				self.net.trainingPass()
				self.updateNetInfo()
				if self.net.getEpochs() % 100 == 0 or self.net.getEpochs() == 1:
					self.drawNet()

	def run(self):
		while(True):
			self.update()
			self.root.update_idletasks()
			self.root.update()

app = App()
app.run()
