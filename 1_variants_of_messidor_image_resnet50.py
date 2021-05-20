# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/gdrive')
import os
os.chdir('/content/gdrive/My Drive/Competitions/kaggle/diabetic-retinopathy-detection/messidor2/')#change dir

"""# New Section"""

#from google.colab import files
#files.upload() #this will prompt you to update the json

#!pip install -q kaggle
#!pip install -q kaggle-cli
#!mkdir -p ~/.kaggle
#!cp kaggle.json ~/.kaggle/
#!ls ~/.kaggle
#!chmod 600 /root/.kaggle/kaggle.json  # set permission

#!kaggle competitions download -c diabetic-retinopathy-detection

#!cat train.* > train-full.zip
#!7z x train.zip.01
#!7z x test.zip.001
#!7z x 'IMAGES.zip.001'

# Commented out IPython magic to ensure Python compatibility.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import imagenet_utils
from skimage import data, color, feature

import os
from tqdm import tqdm_notebook as tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
from keras.applications.resnet import ResNet101
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import ResNet152
from keras.applications.inception_v3 import InceptionV3
from keras.applications import MobileNet
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16

!pip install ipython-autotime
# %load_ext autotime

df_train = pd.read_csv('messidor_data.csv')

df_train.head()

targets_series = pd.Series(df_train['adjudicated_dme'])
one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)
print(one_hot_labels)

im_size1 = 128
im_size2 = 128

x_train = []
y_train = []
x_test = []

#!rsync -r "gdrive/My Drive/Competitions/kaggle/diabetic-retinopathy-detection/input/train/" "gdrive/My Drive/Competitions/kaggle/diabetic-retinopathy-detection/input/train_cp"

"""i = 0 
for f, second, third, fourth in tqdm(df_train.values):
    if type(cv2.imread('IMAGES/{}'.format(f)))==type(None):
        cv2.imread('IMAGES/{}'.format(f))
        print(f)
        #continue
    else:
        img = cv2.imread('IMAGES/{}'.format(f))
        label = one_hot_labels[i]
        resizedImage = cv2.resize(img, (im_size1, im_size2))

        x_train.append( resizedImage)
        #x_train.append(resizedImageVec)
        #x_train.append(canny)
        #x_train = np.expand_dims(x_train, axis=-1) 
        y_train.append(label)
        i += 1
np.save('x_train2_messidor2',x_train)
np.save('y_train2_messidor2',y_train)
print('Done')
"""

x_train = np.load('x_train2_messidor2.npy')
y_train = np.load('y_train2_messidor2.npy')

y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.

print(x_train_raw.shape)
print(y_train_raw)

#!pip uninstall scikit-learn
#!pip install scikit-learn==0.19.1

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.1, random_state=1)

num_class = y_train_raw.shape[1]

from keras.applications.resnet50 import ResNet50
from keras.layers import AveragePooling2D

base_model = ResNet50(weights = None, include_top=False, input_shape=(im_size1, im_size2, 3))

# Add a new top layer
x = base_model.output
x = AveragePooling2D(pool_size=(4, 4))(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
#x = Dense(1024, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)
#predictions = Dense(num_class, activation='sigmoid')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
#for layer in base_model.layers:
#    layer.trainable = False

# model.compile(loss='categorical_crossentropy', 
#               optimizer='rmsprop', 
#               metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', 
#               optimizer='Adagrad', 
#               metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', 
#               optimizer='Adam', 
#               metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', 
              optimizer='SGD', 
              metrics=['accuracy'])


callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', verbose=1)]
model.summary()

model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid), verbose=1)

X_train_features = []
X_valid_features = []

feature_network = Model(base_model.input, model.get_layer('flatten_1').output)

X_train_features = feature_network.predict(X_train)  # Assuming you have your images in x
X_valid_features = feature_network.predict(X_valid)  # Assuming you have your images in x

print(X_train_features.shape)
print(X_valid_features.shape)
print(Y_train.shape)
print(Y_valid.shape)

from sklearn import svm
from sklearn.model_selection import GridSearchCV
import time

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
svm_orig = svm.LinearSVC(max_iter=1000, dual=False)
svm_orig = GridSearchCV(svm_orig, param_grid)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)

svm_orig.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(svm_orig)

# Predict on test data
svm_predict_orig = svm_orig.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)

# Linear Kernal Support Vector Machine
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
# param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
classifier = SVC(kernel = 'linear', C = 1e-1,  max_iter=1000, random_state = 1)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)


classifier.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(classifier)

# Predict on test data
svm_predict_orig = classifier.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)

# RBF Support Vector Machine
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
# param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
classifier = SVC(kernel = 'rbf', C = 1e-1,  max_iter=1000, random_state = 0)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)


classifier.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(classifier)

# Predict on test data
svm_predict_orig = classifier.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)

# RBF GBF
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
# param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
classifier = SVC(kernel = 'sigmoid', C = 1e-1,  max_iter=1000, random_state = 0)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)


classifier.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(classifier)

# Predict on test data
svm_predict_orig = classifier.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)

# ID3 or Decision Tree classifier 
from sklearn.tree import DecisionTreeClassifier
import time

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
# param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)


classifier.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(classifier)

# Predict on test data
svm_predict_orig = classifier.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
import time

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
# param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 1)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)


classifier.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(classifier)

# Predict on test data
svm_predict_orig = classifier.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
import time

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
# param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
classifier = GaussianNB()
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)


classifier.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(classifier)

# Predict on test data
svm_predict_orig = classifier.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)

# K Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
import time

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
# param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)


classifier.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(classifier)

# Predict on test data
svm_predict_orig = classifier.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)

# K Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
import time

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
# param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)


classifier.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(classifier)

# Predict on test data
svm_predict_orig = classifier.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)

# K Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
import time

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
# param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)


classifier.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(classifier)

# Predict on test data
svm_predict_orig = classifier.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)

# AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
import time

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
# param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)


classifier.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(classifier)

# Predict on test data
svm_predict_orig = classifier.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)

# MLP Classifier
from sklearn.neural_network import MLPClassifier
import time
import numpy as np

# Hyper parameters
# C penalty parameter of error term. Smaller values -> stronger regularization.
# param_grid = {'C': [1e-1, 1e0], 'max_iter': [500, 1000]}

# Create model and fit to training data. 
# Do grid search CV to find the best hyperparameters
start_time = time.time()
classifier = MLPClassifier(max_iter=50, random_state=0)
Y_Train_Array = np.argmax(Y_train, axis=1)
print(Y_Train_Array.shape)


classifier.fit(X=X_train_features, y=Y_Train_Array)
print("--- %s seconds ---" % (time.time() - start_time))

# Print model with chosen hyperparameters
print(classifier)

# Predict on test data
svm_predict_orig = classifier.predict(X_valid_features)

# Get accuracy
svm_acc_orig = (svm_predict_orig == np.argmax(Y_valid, axis=1)).mean()
print(svm_acc_orig)