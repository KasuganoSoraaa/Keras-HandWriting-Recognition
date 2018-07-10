import cv2
import numpy as np
from keras.models import load_model

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# model from keras_hw.py
model = load_model('my_model.h5')

# my test picture(made by photoshop)
image = cv2.imread('7re.png')
img = rgb2gray(cv2.imread('7re.png'))

img = img.reshape(1, 28, 28, 1).astype('float32')/255
predict = model.predict_classes(img)
print('您的数字为：')
print(predict)

cv2.imshow('image',image)
cv2.waitKey(0)
