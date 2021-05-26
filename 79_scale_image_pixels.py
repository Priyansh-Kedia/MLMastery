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


# Global centering (When image pixels are centered altogether)

# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)

# convert from integers to floats
pixels = pixels.astype('float32')

# calculate global mean
mean = pixels.mean()
print('Mean: %.3f' % mean)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

# global centering of pixels
pixels = pixels - mean

# confirm it had the desired effect
mean = pixels.mean()
print('Mean: %.3f' % mean)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))


# Local centering (When image pixels are centered per channel)

# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)

# convert from integers to floats
pixels = pixels.astype('float32')

# calculate per-channel means and standard deviations
means = pixels.mean(axis=(0,1), dtype='float64')
print('Means: %s' % means)
print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))

# per-channel centering of pixels
pixels -= means

# confirm it had the desired effect
means = pixels.mean(axis=(0,1), dtype='float64')
print('Means: %s' % means)
print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))


# Global standardization

image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)

# convert from integers to floats
pixels = pixels.astype('float32')

# calculate global mean and standard deviation
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))

# global standardization of pixels
pixels = (pixels - mean) / std

# confirm it had the desired effect
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))


# Positive Global standardization
# There may be a desire to maintain the pixel values in the positive domain, 
# perhaps so the images can be visualized or perhaps for the benefit of a 
# chosen activation function in the model.
# A popular way of achieving this is to clip the standardized pixel values 
# to the range [-1, 1] and then rescale the values from [-1,1] to [0,1].

from numpy import clip

image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)

# convert from integers to floats
pixels = pixels.astype('float32')

# calculate global mean and standard deviation
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))

# global standardization of pixels
pixels = (pixels - mean) / std

# clip pixel values to [-1,1]
pixels = clip(pixels, -1.0, 1.0)

# shift from [-1,1] to [0,1] with 0.5 mean
pixels = (pixels + 1.0) / 2.0

# confirm it had the desired effect
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))



# local standardization

# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)

# convert from integers to floats
pixels = pixels.astype('float32')

# calculate per-channel means and standard deviations
means = pixels.mean(axis=(0,1), dtype='float64')
stds = pixels.std(axis=(0,1), dtype='float64')
print('Means: %s, Stds: %s' % (means, stds))

# per-channel standardization of pixels
pixels = (pixels - means) / stds

# confirm it had the desired effect
means = pixels.mean(axis=(0,1), dtype='float64')
stds = pixels.std(axis=(0,1), dtype='float64')
print('Means: %s, Stds: %s' % (means, stds))
