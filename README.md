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
