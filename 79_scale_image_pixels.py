from PIL import Image

# load the image
image = Image.open("sydney_bridge.jpg")

# some details about the image
print(image.format)
print(image.mode)
print(image.size)

# show the image
image.show()