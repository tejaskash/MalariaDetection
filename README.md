# Deep Learning For Medical Images
Using Deep Learning to detect malaria in thin blood smears. Through this repository you can learn how to train your own machine learning model using keras and tensorflow to detect malaria in images of blood smears.


Malaria is typically found in tropical and subtropical climates where the parasites can live. The World Health Organization (WHO) states that, in 2016, there were an estimated 216 million cases of malaria in 91 countries. In the United States, the Centers for Disease Control and Prevention (CDC) report 1,700 cases of malaria annually. Most cases  malaria develop in people who travel to countries where malaria is more common<sup><a href = "https://www.healthline.com/health/malaria" target = "_blank">[1]</a></sup>.


## Download the dataset

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
