import numpy as np
import pandas as pd
import seaborn as sns
import random
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
from sklearn.decomposition import PCA
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore",category=plt.cbook.mplDeprecation)
random.seed(54)


# Read csv file using pandas libray
train=pd.read_csv('heart.csv')


# Show first 5 rows in data
train.head()

# Data shape 303 rows and 14 columns
train.shape


# Total number of null values
train.isnull().sum()


# Data type of each column
train.dtypes

# Visualizations of data
fig=plt.figure(figsize=(16,16))
fig = plt.subplot(531)
sns.histplot(train['age'],kde=True,label=' Age',color='pink')

fig = plt.subplot(532)
x=train['sex'].value_counts()
x.plot(kind='bar',label='sex',color='pink',xlabel='sex',ylabel='Density')

fig = plt.subplot(533)
x=train['cp'].value_counts()
x.plot(kind='bar',label='cp',color='pink',xlabel='cp',ylabel='Density')

fig = plt.subplot(534)
sns.histplot(train['trtbps'],color='lightgreen',kde=True)

fig = plt.subplot(535)
sns.histplot(train['chol'],kde=True,label='Cholestrol',color='green')

fig = plt.subplot(536)
x=train['fbs'].value_counts()
x.plot(kind='bar',label='fbs',xlabel='fbs',color='lightgreen',ylabel='Density')

fig = plt.subplot(537)
x=train['restecg'].value_counts()
x.plot(kind='bar',label='restecg',xlabel='restecg',color='lightblue',ylabel='Density')

fig = plt.subplot(538)
sns.histplot(train['thalachh'],kde=True, label='thalachh')


fig = plt.subplot(539)
x=train['exng'].value_counts()
x.plot(kind='bar',label='exng',xlabel='exng',ylabel='Density')

fig = plt.subplot(5,3,10)
sns.histplot(train['oldpeak'],color='pink', label='oldpeak')

fig = plt.subplot(5,3,11)
x=train['slp'].value_counts()
x.plot(kind='bar',label='slp',xlabel='slp',color='pink',ylabel='Density')

fig = plt.subplot(5,3,12)
x=train['caa'].value_counts()
x.plot(kind='bar',label='caa',xlabel='caa',color='pink',ylabel='Density')


fig = plt.subplot(5,3,13)
x=train['thall'].value_counts()
x.plot(kind='bar',label='thall',xlabel='thall',color='green',ylabel='Density')

fig = plt.subplot(5,3,14)
x=train['output'].value_counts()
x.plot(kind='bar',label='output',xlabel='output',color='green',ylabel='Density')


plt.show()

# Heat map for attributes
fig=plt.figure(figsize=(10,10))
sns.heatmap(train.corr(),cmap='coolwarm')


# We have seen clearly that chest pain type are a positive relationship with heart rate.

fig=plt.figure(figsize=(16,16))
sns.catplot(x="cp",y='thalachh',hue='output',kind='box',data=train)

# Output labels are unbalanced.

fig=plt.figure(figsize=(5,5))
sns.barplot(x=train['output'],y=train['thalachh'])

# Preprocessing


# Labels are unbalanced so we use the over- resampling method to solve this problem. Thus, each label has the same sample size.
def Resample(data):
    label_0=data[data.output==0] #0
    label_1=data[data.output==1] #1
   
    # upsample minority
    label_0_upsampled = resample(label_0,
                              replace=True, # sample with replacement
                              n_samples=len(label_1), # match number in majority class
                              random_state=27) # reproducible results
    
    # combine majority and upsampled minority
    upsampled = pd.concat([label_1, label_0_upsampled])
    return upsampled


# Call the resample function
train=Resample(train)



# The new shape of data

x=train['output'].value_counts().values
plot=sns.barplot(["0","1"],x)
plot.set(xlabel='Output', ylabel='Number of Data')
plt.show()

# Using z-normalization in order to scaling the values

def normalize(x):

  z=(x-x.mean())/x.std()

  return z


# Dropping output labels for normalization

labels=train['output']
train=train.drop(['output'],axis=1)
train=normalize(train)


# # Cross Validation

# Data split into 80% train and 20%

x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=.2, random_state=1,stratify=labels)


# # Neural Network Model


# Create sequential model

model=keras.Sequential([
     layers.Dense(32,activation='relu',input_shape=[13]),
     layers.Dense(64,activation='relu'),
     layers.Dense(128,activation='relu'),
     layers.Dense(128,activation='relu'),
     layers.Dense(1)

])

# Compiling model using adam optimizer, loss function and accuracy metric

model.compile( 
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss='mae',
    metrics=['accuracy']
)


# Stop training when accuracy is not improved 20 consecutive times

early_stopping= keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='auto',
                                    patience=20,restore_best_weights=True)


# Reducing learning rate when a metric has stopped improving 5 consecutive times

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',mode='auto',factor=0.5,patience=5)


# Fitting model
model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=28,epochs=200,callbacks=[early_stopping,lr_scheduler])

# Evaluating on the test data
model.evaluate(x_test,y_test,verbose=2)


# Predicted label for performance metrics
y_pred=model.predict(x_test).astype("int32")

# Confusion Matrix

cf=confusion_matrix(y_test,y_pred)
sns.heatmap(cf, annot=True).set(xlabel='Actual values',ylabel='Predict values')


# Recal Result
recall_score(y_test,y_pred,average='macro')


# F1 score Result
f1_score(y_test,y_pred,average='macro')


# Precision Result
precision_score(y_test,y_pred,average='macro')
