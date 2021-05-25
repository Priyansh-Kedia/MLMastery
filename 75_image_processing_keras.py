# load image with keras
from keras.preprocessing.image import load_img

image = load_img("bondi_beach.jpg")

# report details about the image
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
# show the image
img.show()