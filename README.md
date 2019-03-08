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
  

