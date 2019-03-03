from imutils import build_montages
from imutils import paths
from random import shuffle
from cv2 import putText
from cv2 import cvtColor
from cv2 import imread
from cv2 import FONT_HERSHEY_SIMPLEX
from cv2 import COLOR_BGR2RGB
from cv2 import imwrite
from cv2 import imshow

imagePaths = list(paths.list_images('cell_images/Parasitized'))
shuffle(imagePaths)
imagePaths = imagePaths[:4]

results = []

for p in imagePaths:
    orig = imread(p)
    image = cvtColor(orig,COLOR_BGR2RGB)
    putText(orig,"Parasitized",(3,20),FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    results.append(orig) 

pm = build_montages(results, (128, 128), (4, 4))[0]

imagePaths = list(paths.list_images('cell_images/Uninfected'))
shuffle(imagePaths)
imagePaths = imagePaths[:4]

results = []

for p in imagePaths:
    orig = imread(p)
    image = cvtColor(orig,COLOR_BGR2RGB)
    putText(orig,"Uninfected",(3,20),FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    results.append(orig) 

um = build_montages(results, (128, 128), (4, 4))[0]

imwrite('Parasitzed Montage',pm)
imwrite('Uninfected Montage',um)

imshow(pm,'PM')
imshow(um,'UM')




