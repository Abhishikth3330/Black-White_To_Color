import numpy as np
import cv2

protxt = './colorization_deploy_v2.prototxt'
model = './colorization_release_v2.caffemodel'
points = './pts_in_hull.npy'
image = './flower1.jpg'

net = cv2.dnn.readNetFromCaffe(protxt, model)
pts = np.load(points)

# ab channel - 1x1 convolutions and add them to model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId('conv8_313_rh')
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype('float32')]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]


# input the image
# scale and convert to float type with dnn model
#image conveted to LAB format
image = cv2.imread(image)
scaled = image.astype('float32') / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)


# resizing the file 
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -=50


# L channel
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")


# original and Colorized image
cv2.imshow("Original Image", image)
cv2.imshow("Colorized Image", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()
