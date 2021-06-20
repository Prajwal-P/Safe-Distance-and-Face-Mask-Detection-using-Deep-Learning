# Prerequisits: 1)download required packages mentioned in requirements.txt
# 2) Setup kaggle using the link https://adityashrm21.github.io/Setting-Up-Kaggle/
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from imutils import paths
import numpy as np
import os
import kaggle

def load():
    path='./dataset'
    #give the above path relative to dataset folder in your system i.e. this is the path where already downloaded dataset exists in my system.

    # if the image dataset is not already available in the above mentioned path,
    # the dataset from kaggle will be downloaded using below 4 lines of code.
    if not os.path.isdir(path):
        print("[INFO] Downloading dataset from kaggle....")

    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images('./dataset'))
    data = []
    labels = []
    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the input image (224x224) and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)  # converts pixel values between -1 to 1

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)  # converts 1-D vector to 2-D matrix

    return data,labels
