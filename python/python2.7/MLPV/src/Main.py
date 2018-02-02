import MLPV
import GUI

class Main:
	net = MLPV.Net()
	gui = GUI.Gui()

	def __init__(self):
		self.data = []

	def main():
		net.initNet()

	def run():
		while(True):
			if gui.getRestart == True:
				net.initNet();

Main()
