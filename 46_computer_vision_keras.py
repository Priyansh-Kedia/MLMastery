from numpy import asarray
from PIL import Image
from sklearn.preprocessing import StandardScaler

image = Image.open("bondi_beach.jpg")
pixels = asarray(image)

print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

# Converting int to float
pixels = pixels.astype("float32")

scaler = StandardScaler()
pixels = scaler.fit_transform(pixels)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))


# Normalising the values to be in range of 0 - 1
pixels = pixels / 255.0

print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))