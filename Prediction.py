import cv2
import numpy as np
from cnn.resnet import ResNet
from keras.models import load_model
from keras.models import model_from_json
import keras.models

from keras.models import Sequential
from keras.preprocessing import image
from keras.optimizers import SGD
import tensorflow as tfSGD

'''init_lr= 1e-1
opt = SGD(lr=init_lr, momentum=0.9)
model = Sequential()
model = ResNet.build(64, 64, 3, 2, (3, 4, 6),
	(64, 128, 256, 512), reg=0.0005)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])'''

model = Sequential()
with open('mdm.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('mdm.h5')


img = image.load_img('test/tester.png', target_size=(64,64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


pred = model.predict(x)
pred = pred.argmax(axis=1)[0]
print(pred)
