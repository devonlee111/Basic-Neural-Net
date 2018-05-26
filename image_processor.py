import os
import sys
from fractions import Fraction
from PIL import Image

def printHelp():
	print "Help"
	print "USAGE:"
	print "python image_processor.py <infile> <outfile> <compression rate/new size>"
	print "Optional compression rate argument should be given as decimal from 0 - 1. Image will be reduced to specified percentage of original before data conversion."
	print "New size should be smaller than original size and be in the form of width,height."
numArgs = len(sys.argv)

if numArgs < 3:
	print "Not Enough Arguments\n"
	printHelp()
	sys.exit()

elif numArgs > 4:
	print "Too many arguments\n"
	printHelp()
	sys.exit()

compress = 0
compressionRate = None

if numArgs == 4:
	if "," in sys.argv[3]:
		size = sys.argv[3].split(",")

		if len(size) > 2:
			print "Invalid new size"
			sys.exit()

		compressionRate = map(int, size)
		compress = 1

	else:
		try:
			compressionRate = float(sys.argv[3])

			if compressionRate >= 1 or compressionRate <= 0:
				print "Compression rate must be from 0 - 1."
				sys.exit()

			compress = 2

		except:
			print "Encountered Unexpected Argument"
			sys.exit()

path = sys.argv[1]
outFile = open(sys.argv[2], "w+")

try:
	image = Image.open(path)

	if compress == 1:
		if compressionRate[0] > image.size[0] or compressionRate[1] > image.size[1]:
			print "New image size must be less than old image size."
			sys.exit()

		elif compressionRate[0] <= 0 or compressionRate[1] <= 0:
			print "New image size must have a positive value."
			sys.exit()

		image = image.resize((compressionRate[0], compressionRate[1]))

	elif compress == 2:
		image = image.resize((int(image.size[0] * compressionRate), int(image.size[1] * compressionRate)))

	size = image.size
	width = size[0]
	height = size[1]
	newImage = image.convert('RGB')

	for x in range(0, width):
		for y in range(0, height):
			pix = newImage.getpixel((x,y))
			red = pix[0]
			green = pix[1]
			blue = pix[2]
			grey = (red + green + blue) / 3
			wb = 0

			if grey > 127:
				wb = 1

			if x == 0 and y == 0:
				outFile.write(str(wb))

			else:
				outFile.write("," + str(wb))

except:
	width = -1
	height = -1
	numImages = 0

	numImages = sum([len(files) for r, d, files in os.walk(path)])
	outFile.write(str(numImages) + "\n")

	for path, subdirs, files in os.walk(path):
		for name in files:
			label = path.rsplit('/', 1)[-1]
			filePath = os.path.join(path, name)
			image = Image.open(filePath)

			if compress == 1:
				if compressionRate[0] > image.size[0] or compressionRate[1] > image.size[1]:
					print "New image size must be less than old image size."
					sys.exit()

				elif compressionRate[0] <= 0 or compressionRate[1] <= 0:
					print "New image size must have a positive value."
					sys.exit()

				image = image.resize((compressionRate[0], compressionRate[1]))

			elif compress == 2:
				image = image.resize((int(image.size[0] * compressionRate), int(image.size[1] * compressionRate)))

			if width == -1 and height == -1:
				size = image.size
				width = size[0]
				height = size[1]

			else:
				newWidth, newHeight = image.size

				if newWidth != width or newHeight != height:
					print "Images are not the same size\nCanceling image processing..."
					sys.exit()

			for x in range(0, width):
				for y in range(0, height):
					pix = image.convert('RGB').getpixel((x,y))
					red = pix[0]
					green = pix[1]
					blue = pix[2]
					grey = (red + green + blue) / 3
					wb = 0

					if grey > 127:
						wb = 1
			
					if x == 0 and y == 0:
						outFile.write(str(wb))
					else:
						outFile.write("," + str(wb))
		
			numImages += 1
			outFile.write(":" + label + "\n")

outFile.close()
