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

### Loading the dataset's name and results text file 
```python
df_test=pd.read_csv(r'C:\Users\User1\OneDrive\Desktop\CovidProject\test.txt',sep=" ")
df_test.head()
```
![.](https://github.com/ShantanuShastri7/covid_prediction/blob/media/1.PNG)

### Giving column names to the data
```python
df_test.columns=["id","file name","class","data source"]
df_test.head()
```
![.](../media/2.PNG?raw=true)

```python
df_train=pd.read_csv(r'C:\Users\User1\OneDrive\Desktop\CovidProject\train.txt',sep=" ")
df_train.columns=["id","file name","class","data source"]
```

### Checking for null values 
```python
df_train.isnull().sum()
```
![.](../media/3.PNG?raw=true)

```python
df_test.isnull().sum()
```
![.](../media/4.PNG?raw=true)

```python
df_train["class"].unique()
```
![.](../media/5.PNG?raw=true)

```python
df_train["class"].value_counts()
```
![.](../media/6.PNG?raw=true)

```python
df_test["class"].value_counts()
```
![.](../media/7.PNG?raw=true)

## 1-Transfer Learning on VGG-16 without Data Augmentation

### Making another data frame with all the negative samples
```python
df_train_negative=df_train[df_train["class"]=="negative"]
df_train_negative.shape
```
![.](../media/8.PNG?raw=true)

```python
df_train.drop(df_train[df_train["class"]=="negative"].index,inplace=True)
df_train["class"].value_counts()
```
![.](../media/9.PNG?raw=true)

### Randomly sampling data from negative samples to equal the number of positive samples 
```python
df_train_negative=df_train_negative.sample(n=2158,replace=False,random_state=5)
df_train_negative["class"].value_counts()
```
![.](../media/9.PNG?raw=true)

### Concatinating the positive and negative samples 
```python
df_train_reduced=pd.concat([df_train,df_train_negative],axis=0)
df_train_reduced["class"].value_counts()
```
![.](../media/10.PNG?raw=true)

```python
df_train_reduced=df_train_reduced.sample(frac=1)
```

### Re-shuffling positve and negative samples 
```python
df_train=df_train_reduced
df_train["class"].value_counts()
```
![.](../media/11.PNG?raw=true)

### Dropping unnecessary columns from the data set
```python
df_test.drop(["id","data source"],axis=1,inplace=True)
df_train.drop(["id","data source"],axis=1,inplace=True)
df_train.head()
```
![.](../media/12.PNG?raw=true)

```python
df_test.head()
```
![.](../media/13.PNG?raw=true)

### Importing the VGG-16 model along with the weights for transfer learning 
```python
from tensorflow.keras.applications.vgg16 import VGG16
base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')
```
```python
base_model.summary()
```
![.](../media/14.PNG?raw=true)

### Setting the training of all the layers in the base model to False and using the pre trained weights
```python
for layer in base_model.layers:
    layer.trainable = False
```

### Copying the files from the training data folder to 2 different folders names 'positive' and 'negative' under the main training folder 
```python
s="C:\\Users\\User1\\OneDrive\\Desktop\\CovidProject\\train\\"

import shutil, os
for (f,c) in zip(df_train["file name"],df_train["class"]):
        if c=="positive":
            shutil.copy(s+f, 'C:\\Users\\User1\\OneDrive\\Desktop\\CovidProject\\train_final\\positive')
        else:
            shutil.copy(s+f, 'C:\\Users\\User1\\OneDrive\\Desktop\\CovidProject\\train_final\\negative')
```

### Re-scaling the RGB values between 0-1 for faster training and better accuracy
```python
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
valid_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_dir="C:\\Users\\User1\\OneDrive\\Desktop\\CovidProject\\train_final"
validation_dir="C:\\Users\\User1\\OneDrive\\Desktop\\CovidProject\\validation"
```

### Initialising Image generator for feeding training and validation image samples to the model
```python
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 50, class_mode = 'binary', target_size = (224, 224))
```
![.](../media/15.PNG?raw=true)

```python
validation_generator = valid_datagen.flow_from_directory( validation_dir,  batch_size = 50, class_mode = 'binary', target_size = (224, 224))
```
![.](../media/16.PNG?raw=true)

### Adding layers in the end to classify images as positive and negative (Binary Classification) using sigmoid function
```python

# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
```
```python
model.summary()
```
![.](../media/17.PNG?raw=true)

### Training
```python
vgghist = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 87, epochs = 10)
```
![.](../media/18.PNG?raw=true)

### Plotting Training and Validation accuracy and loss
```python
train_acc= vgghist.history['acc']
train_loss = vgghist.history['loss']
valid_acc= vgghist.history['val_acc']
valid_loss= vgghist.history['val_loss']
```
```python
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(valid_acc,label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(2,2,2)
plt.plot(train_loss, label='Training Loss')
plt.plot(valid_loss,label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
```
![.](../media/19.PNG?raw=true)

#### The accuracy begins with 77% and goes up to 95.67% in the last epoch and on the test set we have a max. accuracy of 96.5%.  Because of Transfer Learning we are able to get such high accuracies even without Data Augmentation
```python
model.save("model_vgg.h5")
```

## 2-Transfer Learning on VGG-16 with Data Augmentation

### Loading the data set again with all samples
```python
df_train_VGG_A=pd.read_csv(r'C:\Users\User1\OneDrive\Desktop\CovidProject\train.txt',sep=" ")
df_train_VGG_A.columns=["id","file name","class","data source"]
df_train_VGG_A.drop(["id","data source"],axis=1,inplace=True)
df_train_VGG_A.head()
```
![.](../media/20.PNG?raw=true)

### Giving ImageDataGenerator parameters for doing Data Augmentation
1. Rotation Range of 20 degrees which is close to human error while taking photos
2. width_shift_range 0f .1 for minor horizontal shifts 
3. height_shift_range of .1 for minor vertical shifts
4. zoom range of .1 for minor zoomed in images 

 ```python
 datagen = ImageDataGenerator(rotation_range = 20, width_shift_range = 0.1, height_shift_range = 0.1, zoom_range = 0.1)
 df_train_VGG_A.drop(df_train_VGG_A[df_train_VGG_A["class"]=="negative"].index,inplace=True)
 ```
 
 ### Giving Image Generator images in batches of 1000 and saving the generated images

( Because my RAM was over flowing after 1K images at a go)

```python
df_train_VGG_A_0_1000=df_train_VGG_A[:1000]
df_train_VGG_A_1000_2000=df_train_VGG_A[1000:2000]
df_train_VGG_A_2000_2158=df_train_VGG_A[2000:2159]
```
```python
t='C:\\Users\\User1\\OneDrive\\Desktop\\CovidProject\\train\\'
for f in df_train_VGG_A_0_1000['file name']:
    img=load_img(t+f)
    x=img_to_array(img)
    x=x.reshape((1,)+x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size = 1,
                          save_to_dir ='C:\\Users\\User1\\OneDrive\\Desktop\\CovidProject\\train_augmented', 
                          save_prefix =(f[:len(f)-5]+str(i)), save_format ='jpeg'):
            i += 1
            if i > 5:
                break
 ```
 ### Importing the VGG-16 model along with the weights for transfer learning  
 ```python
 from tensorflow.keras.applications.vgg16 import VGG16

base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')
```
```python
base_model.summary()
```
![.](../media/21.PNG?raw=true)

### Setting the training of all the layers in the base model to False and using the pre trained weights
```python
for layer in base_model.layers:
    layer.trainable = False
```

### Re-scaling the RGB values between 0-1 for faster training and better accuracy
```python
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
valid_datagen = ImageDataGenerator( rescale = 1.0/255. )
```

### Initialising Image generator for feeding training and validation image samples to the model
```python
train_dir="C:\\Users\\User1\\OneDrive\\Desktop\\CovidProject\\train_final_DA"
validation_dir="C:\\Users\\User1\\OneDrive\\Desktop\\CovidProject\\validation"
```
```python
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, class_mode = 'binary', target_size = (224, 224))
```
![.](../media/22.PNG?raw=true)

```python
validation_generator = valid_datagen.flow_from_directory(validation_dir,  batch_size = 20, class_mode = 'binary', target_size = (224, 224))
```
![.](../media/23.PNG?raw=true)

### Adding layers in the end to classify images as positive and negative (Binary Classification) using sigmoid function
```python

# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model_vgg_da = tf.keras.models.Model(base_model.input, x)

model_vgg_da.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
```
```python
model_vgg_da.summary()
```
![.](../media/24.PNG?raw=true)

### Training
```python
vgg_A_hist = model_vgg_da.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 117, epochs = 10)
```
![.](../media/25.PNG?raw=true)

### Plotting Training and Validation accuracy and loss
```python
train_acc= vgg_A_hist.history['acc']
train_loss = vgg_A_hist.history['loss']
valid_acc= vgg_A_hist.history['val_acc']
valid_loss= vgg_A_hist.history['val_loss']
```
```python
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(valid_acc,label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(2,2,2)
plt.plot(train_loss, label='Training Loss')
plt.plot(valid_loss,label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
```
![.](../media/26.PNG?raw=true)

#### Accuracy on the training set started from 85% and went up to 96.67%, on the test set we had a max. accuracy of 87.25%.  As per the theory we should have got an increase in the accuracy after data augmentation, but that wasn't the case. My prediction being that-We generated around 10K positive samples from 2158 samples so the model might have over fitted and hence a reduced accuracy
```python
model_vgg_da.save("model_vgg_da.h5")
```

## 3-Training a CNN from scratch without Data Augmentation

### Iniatialising a sequential model
```python
model_cnn=Sequential()
```
### Adding the required layers
```python
    model_cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224,224,3)))
    model_cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.25))

    model_cnn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model_cnn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.25))

    model_cnn.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model_cnn.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.25))

    model_cnn.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model_cnn.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.25))
    
    model_cnn.add(Flatten())
    model_cnn.add(Dense(512, activation='relu'))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(1, activation='sigmoid'))
```
```python
model_cnn.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
```
```python
model_cnn.summary()
```
![.](../media/27.PNG?raw=true)

### Re-scaling the RGB values between 0-1 for faster training and better accuracy
```python
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
valid_datagen = ImageDataGenerator( rescale = 1.0/255. )
```

### Initialising Image generator for feeding training and validation image samples to the model
```python
train_dir="C:\\Users\\User1\\OneDrive\\Desktop\\CovidProject\\train_final"
validation_dir="C:\\Users\\User1\\OneDrive\\Desktop\\CovidProject\\validation"
```
```python
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 50, class_mode = 'binary', target_size = (224, 224))
```
![.](../media/28.PNG?raw=true)

```python
validation_generator = valid_datagen.flow_from_directory(validation_dir,  batch_size = 20, class_mode = 'binary', target_size = (224, 224))
```
![.](../media/29.PNG?raw=true)

### Training
```python
trialmodel = model_cnn.fit(train_generator, validation_data = validation_generator, steps_per_epoch=86, epochs = 10)
```
![.](../media/30.PNG?raw=true)

### Plotting Training and Validation accuracy and loss
```python
train_acc=trialmodel.history['acc']
train_loss = trialmodel.history['loss']
valid_acc=trialmodel.history['val_acc']
valid_loss=trialmodel.history['val_loss']
```
```python
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(valid_acc,label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(2,2,2)
plt.plot(train_loss, label='Training Loss')
plt.plot(valid_loss,label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
```
![.](../media/31.PNG?raw=true)

#### Accuracy on the training set started from 65.64% and went up to 93% and on the test set max. accuracy was 91%
```python
model_cnn.save("model_cnn.h5")
```

![.](../media/32.PNG?raw=true)
## Best Accuracy was given by VGG-16 without using Data Augmentation
