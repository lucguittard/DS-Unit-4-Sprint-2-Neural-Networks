#!/usr/bin/env python
# coding: utf-8

# <img align="left" src="https://lever-client-logos.s3.amazonaws.com/864372b1-534c-480e-acd5-9711f850815c-1524247202159.png" width=200>
# <br></br>
# 
# # Neural Network Framework (Keras)
# 
# ## *Data Science Unit 4 Sprint 2 Assignmnet 3*
# 
# ## Use the Keras Library to build a Multi-Layer Perceptron Model on the Boston Housing dataset
# 
# - The Boston Housing dataset comes with the Keras library so use Keras to import it into your notebook. 
# - Normalize the data (all features should have roughly the same scale)
# - Import the type of model and layers that you will need from Keras.
# - Instantiate a model object and use `model.add()` to add layers to your model
# - Since this is a regression model you will have a single output node in the final layer.
# - Use activation functions that are appropriate for this task
# - Compile your model
# - Fit your model and report its accuracy in terms of Mean Squared Error
# - Use the history object that is returned from model.fit to make graphs of the model's loss or train/validation accuracies by epoch. 
# - Run this same data through a linear regression model. Which achieves higher accuracy?
# - Do a little bit of feature engineering and see how that affects your neural network model. (you will need to change your model to accept more inputs)
# - After feature engineering, which model sees a greater accuracy boost due to the new features?

# In[7]:


# Import Boston Housing dataset -- stay civil 
from keras.datasets import boston_housing as dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


# In[154]:


import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


# In[26]:


# Make a dataframe from the imported data

X = pd.DataFrame(x_train)
y = pd.DataFrame(y_train)

dataset = pd.concat([X, y], axis=1)
df = pd.DataFrame(data = dataset)


# In[173]:


# Let's scale the dataset
names = df.columns# Create the Scaler object
scaler = preprocessing.MinMaxScaler()# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)


# In[174]:


print(scaled_df.shape)
scaled_df.head()


# In[175]:


y = scaled_df.iloc[:, 13].values
X = scaled_df.iloc[:, 0:13].values
print(X[0])
print(y[0])


# In[176]:


y = scaled_df.values[:,-1]
X = scaled_df.values[:, 0:13]
print(X[0])
print(y[0])


# In[177]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

np.random.seed(1)


# In[184]:


# Regression Example With Boston Dataset: Standardized
# source: https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

# Define base model
def baseline_model():
    # Create a model
    model = Sequential()
    model.add(Dense(25, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal'))
    
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


# In[185]:


scores = model.evaluate(X,y)
print(f"{model.metrics_names[1]}: {scores[1]*100}")


# In[95]:


import matplotlib.pyplot as plt

history = model.fit(X, y, batch_size=10, epochs=5, verbose=False, validation_split=.1)
loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')


# ## Use the Keras Library to build an image recognition network using the Fashion-MNIST dataset (also comes with keras)
# 
# - Load and preprocess the image data similar to how we preprocessed the MNIST data in class.
# - Make sure to one-hot encode your category labels
# - Make sure to have your final layer have as many nodes as the number of classes that you want to predict.
# - Try different hyperparameters. What is the highest accuracy that you are able to achieve.
# - Use the history object that is returned from model.fit to make graphs of the model's loss or train/validation accuracies by epoch. 
# - Remember that neural networks fall prey to randomness so you may need to run your model multiple times (or use Cross Validation) in order to tell if a change to a hyperparameter is truly producing better results.

# In[2]:


# Import the data and appropriate libraries
get_ipython().system('pip install -U tensorflow>=1.8.0')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)


# In[3]:


plt.imshow(x_train[0]) # Your Jordans are... green?


# In[4]:


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# In[5]:


# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')


# In[6]:


model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, 
                                 padding='same', activation='relu',
                                 input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2,
                                 padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))# Take a look at the model summary

model.summary()


# In[7]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


# In[8]:


model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(x_valid, y_valid))

# and watch as the accuracy improves


# In[12]:


score = model.evaluate(x_test, y_test, verbose=0)


# ## Stretch Goals:
# 
# - Use Hyperparameter Tuning to make the accuracy of your models as high as possible. (error as low as possible)
# - Use Cross Validation techniques to get more consistent results with your model.
# - Use GridSearchCV to try different combinations of hyperparameters. 
# - Start looking into other types of Keras layers for CNNs and RNNs maybe try and build a CNN model for fashion-MNIST to see how the results compare.
