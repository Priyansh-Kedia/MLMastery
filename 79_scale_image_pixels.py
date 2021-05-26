from PIL import Image

# load the image
image = Image.open("sydney_bridge.jpg")

# some details about the image
print(image.format)
print(image.mode)
print(image.size)

# show the image
image.show()

# Normalize the pixel values
from numpy import asarray

pixels = asarray(image)

# Some details about pixels
print("Data Type %s" % pixels.dtype)
print("Min: %.3f, Max: %.3f" % (pixels.min(), pixels.max()))

# convert values from integers to floats
pixels = pixels.astype("float32")

# normalize the values in the range of 0-1
pixels /= 255

# confirm the normalization
print("Min: %.3f, Max: %.3f" % (pixels.min(), pixels.max()))