import threading
import MLPV
import GUI

class Main:
	#net = MLPV.Net()
	#gui = GUI.Gui()

	def __init__(self):
		self.data = []
		self.net = MLPV.Net()
		self.gui = GUI.Gui()
		
main = Main()
