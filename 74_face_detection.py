# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN


# load image from file
pixels = pyplot.imread('street.jpg')

# create the detector, using default weights
detector = MTCNN()

# detect faces in the image
faces = detector.detect_faces(pixels)

# plot the image
pyplot.imshow(pixels)

# get the context for drawing boxes
ax = pyplot.gca()

# get coordinates from the first face
x, y, width, height = faces[0]['box']

# create the shape
rect = Rectangle((x, y), width, height, fill=False, color='red')

# draw the box
ax.add_patch(rect)

# show the plot
pyplot.show()