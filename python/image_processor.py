import os
import sys
from PIL import Image

numArgs = len(sys.argv)

if numArgs < 3:
	print "Not Enough Arguments\n"
	sys.exit()

elif numArgs == 4:
	print "Not Enough Arguments\n"
	sys.exit()

elif numArgs > 5:
	print "Too many arguments\n"
	sys.exit()

path = sys.argv[1]
outFile = open(sys.argv[2], "w+")

try:
	image = Image.open(path)
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

	if len(sys.argv) == 4:
		width = sys.argv[2]
		height = sys.argv[3]

	numImages = sum([len(files) for r, d, files in os.walk(path)])
	outFile.write(str(numImages) + "\n")

	for path, subdirs, files in os.walk(path):
		for name in files:
			label = path.rsplit('/', 1)[-1]
			filePath = os.path.join(path, name)
			image = Image.open(filePath)
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
