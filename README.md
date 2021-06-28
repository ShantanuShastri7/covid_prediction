# Predicting Covid 19 using chest X-rays

### Context
Covid 19 has been cause of most deaths around the globe since the last one and a half years. COVID-19 is a disease caused by a new strain of coronavirus. 'CO' stands for corona, 'VI' for virus, and 'D' for disease. Formerly, this disease was referred to as '2019 novel coronavirus' or '2019-nCoV.'  

Current evidence suggests that the virus spreads mainly between people who are in close contact with each other, typically within 1 metre (short-range). A person can be infected when aerosols or droplets containing the virus are inhaled or come directly into contact with the eyes, nose, or mouth.

Detection of Covid-19 is done using RT-PCR tests, the major downside of this test is that it takes a long amount of time to get processed which thereby increases the chances of patient's degradation in health.
There is another process by which doctors are able to detect the presence of Corona virus, it is done using X-Rays. But during these times the availability of a doctor is also difficult to each and every person. So using DNN to train and predict Covid-19 will be of a great use.

### Data Set
The data set was taken from kaggle, the link to which is 
[COVIDx CXR-2 Dataset](https://www.kaggle.com/andyczhao/covidx-cxr2)  
The data set had 2 folders 
1. Train 
2. Test 

Train folder had 15971 X-ray images belonging to both the positive and negative classes, out of which 2158 were Covid-19 positive and 13792 were Covid-19 negative  
Test folder has 400 X-ray images, out of which 200 were Covid-19 positive and the other 200 were Covid-19 negative 

### Methodology
The data set is imbalanced, i.e. it has 2158 positive examples and 13792 negative examples.  In such a case we have two solutions, either 
1. Under sampling the data (taking same number of positive and negative samples, i.e. 2158 positive and 2158 negative)  
2. Over sampling the data (generating new positive samples in order to match the amount of negative samples)

When it comes to CNNs we have two options
1. Train our own model from scratch
2. Apply Transfer Learning on a pre trained model. 

First we'll take VGG-16 which is pre trained model that won the ImageNet challenge, the architecture is given below

![VGG16](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png) 

Next we'll make a CNN of our own and train it from scratch on our data set, I'll be sharing the architecture in the code 

So to check for accuracies in different model and pre processing, we will be training 3 models in total 

1. Transfer Learning on VGG-16 without Data Augmentation
2. Transfer Learning on VGG-16 with Data Augmentation
3. Training a CNN from scratch without Data Augmentation

#### For the accuracy of the test set, I'll be using my test images on the validation parameter instead of dividing my training data into two parts.

### Importing required libraries

```python
import numpy as np
import pandas as pd
import warnings 
from sklearn.model_selection import train_test_split
import os 
import zipfile 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions 
import itertools 
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
```

