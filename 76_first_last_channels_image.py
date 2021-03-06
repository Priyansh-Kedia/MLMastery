# Channels first is when image is represented as 
# [channels][rows][columns]
# Channels last is when image is represented as 
# [rows][colums][channels]

from numpy import expand_dims
from numpy import asarray
from PIL import Image

# load the image
img = Image.open('penguin_arade.jpg')

# convert the image to grayscale
img = img.convert(mode='L')

# convert to numpy array
data = asarray(img)
print(data.shape)

# add channels first
# Insert a new axis that will appear at the axis position in the expanded array shape.
# Position in the expanded axes where the new axis (or axes) is placed.
data_first = expand_dims(data, axis=0)
print(data_first.shape)

# add channels last
data_last = expand_dims(data, axis=2)
print(data_last.shape)


# change channels in image

from numpy import moveaxis
from numpy import asarray
from PIL import Image

# load the color image
img = Image.open('penguin_arade.jpg')

# convert to numpy array
data = asarray(img)
print(data.shape)

# change channels last to channels first format
# moveaxis: Moves the axis from source to destination
data = moveaxis(data, 2, 0)
print(data.shape)

# change channels first to channels last format
data = moveaxis(data, 0, 2)
print(data.shape)