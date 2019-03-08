# Deep Learning For Medical Images
### Background
Through this repository you can learn how to train your own machine learning model using keras and tensorflow to detect malaria in images of blood smears. 

Malaria is typically found in tropical and subtropical climates where the parasites can live. The World Health Organization (WHO) states that, in 2016, there were an estimated 216 million cases of malaria in 91 countries. In the United States, the Centers for Disease Control and Prevention (CDC) report 1,700 cases of malaria annually. Most cases  malaria develop in people who travel to countries where malaria is more common<sup><a href = "https://www.healthline.com/health/malaria" target = "_blank">[1]</a></sup>. 

## List Of Packages We Will Need 
* Keras : A deeplearning framework based on tensorflow. To install keras, I suggest the following [guide](https://www.pyimagesearch.com/2016/11/14/installing-keras-with-tensorflow-backend/) by Adrian Rosebrock
* Numpy and Scikit-learn : Packages to make it easier to work with arrays and machine learning install by running
  ```bash
  pip install numpy
  pip install scikit-learn
  ```
* Matplotlib: A plotting library for python
  ```bash
  pip install matplotlib
  ```
* imutils: A package of deeplearning and image processing convenience functions written by Dr. Adrian Rosebrocks
  ```bash
  pip install --upgrade imutils
  ```

## Download the dataset

The dataset consists of 27,558 single cell images with an equal number of infected and uninfected cells. The cells are from 200 patients where three out of every four patients had malaria. Single cells were segmented from images of microscope fields of view. Here are a few images from the dataset. 

![Parasitized](https://github.com/tejaskashinathofficial/MalariaDetection/blob/master/assets/PM.png) ![Uninfected](https://github.com/tejaskashinathofficial/MalariaDetection/blob/master/assets/UM.png)

You can download the dataset <a href="https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip" target="_blank">here</a>.

Or if you're using a distribution of linux with ```wget``` installed, then download the repository for this project, and cd into the Dataset folder.

Then run :

```shell
chmod +x dataset.sh
```
And then to download the dataset:

```shell
./dataset.sh
```

Create the following project structure in folder. The cell_images folder is the unzipped form of the dataset that you just downloaded.
  ```bash
  ├── cell_images
  │   ├── Parasitized
  │   └── Uninfected
  ├── cnn  
  │   ├── config.py
  │   └── resnet.py
  ├── BuildDataset.py
  ├── Prediction.py
  └── TrainModel.py
  ```
  ## Process The Dataset
  
  First, we will define all te directories that we will be working with in the ```config.py``` file inside the cnn folder. This will make it easier to access the filepaths accross all the programs that we will be using in this tutorial. Open up the ```config.py``` file and insert the following code.
  
  ```python
  import os
  # initialize the path to the *original* input directory of images
  ORIG_INPUT_DATASET = "/full/path/to/cell_images/folder/"

  # Name of the directory that will contain the test, train and
  # validation batches after the dataset has been split.
  BASE_PATH = "malaria"

  # make the training, validation, and testing directories
  TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
  VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
  TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

  # define the amount of data that will be used training
  TRAIN_SPLIT = 0.8

  # the amount of validation data will be a percentage of the
  # *training* data
  VAL_SPLIT = 0.1
  ```
  Next, we wil write the code that will sort through our ```cell_images``` directory, shuffle the images and store the required images in the testing, training and validation directories. Open up the ```BuildDataset.py``` file and insert the following code.
  
  ```python
  from cnn import config
from imutils import paths
import random
import shutil
import os

imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

i = int(len(imagePaths)*config.TRAIN_SPLIT)

trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

i = int(len(trainPaths)*config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

datasets = [("training",trainPaths,config.TRAIN_PATH),("validation",valPaths,config.VAL_PATH),("testing",testPaths,config.TEST_PATH)]

for(dType, imagePaths,baseOutput) in datasets:
    print("[INFO] building '{}' split".format(dType))

    if not os.path.exists(baseOutput):
        print("[INFO] 'creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)
    
    for inputPath  in imagePaths:
        filename = inputPath.split(os.path.sep)[-1]

        label = inputPath.split(os.path.sep)[-2]

        labelPath = os.path.sep.join([baseOutput, label])

        if not os.path.exists(labelPath):
            print("[INFO] 'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)
        p = os.path.sep.join([labelPath,filename])
        shutil.copy2(inputPath,p) 
```
Now, open up your terminal and run ``` python BuildDataset.py```.  

## Building the ResNet50 architecture

We will be using the ResNet-50 architecture to train or model. The specifics of ResNet are not covered here, you can find more details in the offical ResNet publication [here](https://arxiv.org/abs/1512.03385). The following implementation of ResNet-50 using keras was made by Dr.Adrian Rosebrock, however you can also use ```keras.applications.resnet.ResNet50```, details on how to implement this can be found [here](https://keras.io/applications/#resnet). Open up the ```resnet.py``` file in the ```cnn``` folder and insert the following code.

```python
# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

class ResNet:
	@staticmethod
	def residual_module(data, K, stride, chanDim, red=False,
		reg=0.0001, bnEps=2e-5, bnMom=0.9):
		# the shortcut branch of the ResNet module should be
		# initialize as the input (identity) data
		shortcut = data

		# the first block of the ResNet module are the 1x1 CONVs
		bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(data)
		act1 = Activation("relu")(bn1)
		conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
			kernel_regularizer=l2(reg))(act1)

		# the second block of the ResNet module are the 3x3 CONVs
		bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(conv1)
		act2 = Activation("relu")(bn2)
		conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
			padding="same", use_bias=False,
			kernel_regularizer=l2(reg))(act2)

		# the third block of the ResNet module is another set of 1x1
		# CONVs
		bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(conv2)
		act3 = Activation("relu")(bn3)
		conv3 = Conv2D(K, (1, 1), use_bias=False,
			kernel_regularizer=l2(reg))(act3)

		# if we are to reduce the spatial size, apply a CONV layer to
		# the shortcut
		if red:
			shortcut = Conv2D(K, (1, 1), strides=stride,
				use_bias=False, kernel_regularizer=l2(reg))(act1)

		# add together the shortcut and the final CONV
		x = add([conv3, shortcut])

		# return the addition as the output of the ResNet module
		return x

	@staticmethod
	def build(width, height, depth, classes, stages, filters,
		reg=0.0001, bnEps=2e-5, bnMom=0.9):
		# initialize the input shape to be "channels last" and the
		# channels dimension itself
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# set the input and apply BN
		inputs = Input(shape=inputShape)
		x = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(inputs)

		# apply CONV => BN => ACT => POOL to reduce spatial size
		x = Conv2D(filters[0], (5, 5), use_bias=False,
			padding="same", kernel_regularizer=l2(reg))(x)
		x = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(x)
		x = Activation("relu")(x)
		x = ZeroPadding2D((1, 1))(x)
		x = MaxPooling2D((3, 3), strides=(2, 2))(x)

		# loop over the number of stages
		for i in range(0, len(stages)):
			# initialize the stride, then apply a residual module
			# used to reduce the spatial size of the input volume
			stride = (1, 1) if i == 0 else (2, 2)
			x = ResNet.residual_module(x, filters[i + 1], stride,
				chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

			# loop over the number of layers in the stage
			for j in range(0, stages[i] - 1):
				# apply a ResNet module
				x = ResNet.residual_module(x, filters[i + 1],
					(1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

		# apply BN => ACT => POOL
		x = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(x)
		x = Activation("relu")(x)
		x = AveragePooling2D((8, 8))(x)

		# softmax classifier
		x = Flatten()(x)
		x = Dense(classes, kernel_regularizer=l2(reg))(x)
		x = Activation("softmax")(x)

		# create the model
		model = Model(inputs, x, name="resnet")

		# return the constructed network architecture
		return model
```

## Train Your Model

Now, using the ResNet50 based model we built above, we will train our model. Since we are dealing with a large dataset, your system might not have enough memory to koad the entire dataset at once, so we will use keras's ```ImageDataGenerator``` class to load the images in batches. Now open up the ```TrainModel.py``` file and insert the following code. 

```python 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.models import model_from_json
from cnn.resnet import ResNet
from cnn import config
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
BS = 32
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# define the # of epochs, initial learning rate and batch size
num_epochs = 50
init_lr= 1e-1
bs = 32
 
# create a function called polynomial decay which helps us decay our 
# learning rate after each epoch

def poly_decay(epoch):
	# initialize the maximum # of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = num_epochs
	baseLR = init_lr
	power = 1.0  # turns our polynomial decay into a linear decay
 
	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
 
	# return the new learning rate
	return alpha

# determine the # of image paths in training/validation/testing directories
totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# initialize the training data augmentation object
# randomly shifts, translats, and flips each training sample
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")
 
# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=True,
	batch_size=bs)
 
# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=bs)
 
# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=bs)

# initialize our ResNet model and compile it
model = ResNet.build(64, 64, 3, 2, (3, 4, 6),
	(64, 128, 256, 512), reg=0.0005)
opt = SGD(lr=init_lr, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# define our set of callbacks and fit the model
callbacks = [LearningRateScheduler(poly_decay)]
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	epochs=num_epochs,
	callbacks=callbacks)

# reset testing generator and use the trained model to make predictions 
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // BS) + 1)
 
# finds the index of the label with largest predicted probability of each
# testing image 

predIdxs = np.argmax(predIdxs, axis=1) 
 
# displays a classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))

# plot the training loss and accuracy
N = num_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

model_json = model.to_json()
with open("mdm.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("mdm.h5")

```
To train your model, open up your terminal and run ```python TrainModel.py```
In the code above, we have created a plot of our accuracies using ```matplotlib``` and saved it as ```plot.png```. Also, we have saved the weights of the model as a ```h5``` file, so we can use it to make predictions in the future.

Here is a visualization of how the loss decreased and accuracy imporved as we trained our model over 50 epochs.

![Plot](https://github.com/tejaskashinathofficial/MalariaDetection/blob/master/assets/plot.png)


## Testing our model and Getting a prediction

The pretrained weights for this model are available in the repository for this project, to get a prediction from our model, open up the ```Prediction.py``` folder and insert the following code. 

```python
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to out input directory of images")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())


print("[INFO] loading pre-trained network...")
model = load_model(args["model"])


imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:16]
 
# initialize our list of results
results = []

for p in imagePaths:
    orig = cv2.imread(p)
    image = cv2.cvtColor(orig,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(64,64))
    image = image.astype('float')/255.0
    image = np.expand_dims(image,axis=0)

    pred = model.predict(image)
    pred = pred.argmax(axis=1)[0]
    if pred == 0:
        label = "Parasitized"
    else:
        label = "Uninfected"
    if pred == 0:
        color = (0,0,255)
    else:
        color = (0,255,0)
    orig = cv2.resize(orig,(128,128))
    cv2.putText(orig,label,(3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    results.append(orig)

montage = build_montages(results, (128, 128), (4, 4))[0]
 
# show the output montage
cv2.imshow("Results", montage)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
Then open up your terminal and run ```python Prediction.py --images malaria/testing  --model mdm1.model```

The code will load your model, and make a prediction on a batch of 16 random images from your testing folder.

Here was my output,

![output](https://github.com/tejaskashinathofficial/MalariaDetection/blob/master/assets/output.png)


