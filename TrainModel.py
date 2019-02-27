from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from imutils import paths
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
from cnn import config
INIT_LR = 1e-1

def poly_decay(epoch):
    maxEpochs = 25
    baseLR = INIT_LR
    power = 1.0

    alpha = baseLR*(1-(epoch/float(maxEpochs)))**power
    return alpha
classifier  = Sequential()

classifier.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128,activation='relu'))

classifier.add(Dense(units = 2,activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

trainAug = ImageDataGenerator(
    rescale = 1/255.0,
    rotation_range=20,
    zoom_range = 0.05,
    height_shift_range = 0.05,
    shear_range = 0.05,
    horizontal_flip = True,
    fill_mode = "nearest")

valAug = ImageDataGenerator(rescale = 1/255.0)

trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode = "categorical",
    target_size = (64,64),
    color_mode = "rgb",
    shuffle = True,
    batch_size = 32)

testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode = "categorical",
    target_size = (64,64),
    color_mode = "rgb",
    shuffle = False,
    batch_size = 32
)
opt = SGD(lr=INIT_LR,momentum=0.9)
callbacks = [LearningRateScheduler(poly_decay)]
classifier.fit_generator(trainGen,steps_per_epoch = totalTrain//32 ,epochs = 25,validation_data = testGen,validation_steps = totalVal//32,callbacks = callbacks)

print("[INFO] Evaluating Network .... ")
testGen.reset()

pred = classifier.predict_generator(testGen,steps = (totalTest//32)+1)

pred = np.argmax(pred,axis = 1)

print(classification_report(testGen.classes,pred,target_names=testGen.class_indices.keys()))

N = 25

plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0,N),H.history["loss"],label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.hispch_skylake-virtual-0
tory["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])