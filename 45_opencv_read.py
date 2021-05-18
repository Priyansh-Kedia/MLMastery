import cv2 as cv


image = cv.imread("one.jpeg")
cv.imshow("image", image)
cv.waitKey(0)
cv.destroyAllWindows()

resized = cv.resize(image, (28,28), interpolation=cv.INTER_AREA)
cv.imshow("resized", resized)
cv.waitKey(0)
cv.destroyAllWindows()