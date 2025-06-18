# from google.colab import drive
# drive.mount('/content/drive')

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, BatchNormalization, concatenate, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Reshape
from keras.callbacks import EarlyStopping
# from keras.layers.core import SpatialDropout2D
from keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
import pandas as pd
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import InceptionResNetV2, EfficientNetV2L, Xception
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
from keras.models import load_model
import itertools
filterwarnings('ignore')
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
train_dir = "/home/kaga/Desktop/mingyang/dataset/Unique2/Betel2/Train"
val_dir = "/home/kaga/Desktop/mingyang/dataset/Unique2/Betel2/Valid"
test_dir = "/home/kaga/Desktop/mingyang/dataset/Unique2/Betel2/Test"
labels = ['Healthy Betel Leaf','Unhealthy Betel Leaf']
label_np=np.array(labels)
import os
import random


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


random_seed = np.random.randint(0, 10000)
print(f"Generated random seed: {random_seed}")


set_seed(random_seed)


model1 =  InceptionResNetV2(include_top=False,input_shape=(299, 299,3), weights='imagenet')
input_shape = (299, 299)


model2 = EfficientNetV2L(include_top=False,input_shape=(299, 299,3), weights='imagenet')
input_shape = (299, 299)


model3 = Xception(include_top=False,input_shape=(299, 299,3), weights='imagenet')
input_shape = (299, 299)


datagen_train = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  rotation_range=2,
                                  vertical_flip=False,
                                  horizontal_flip=True)

datagen_test = ImageDataGenerator(rescale=1./255)

datagen_val=ImageDataGenerator(rescale=1./255)

batch_size = 4

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True)

generator_val = datagen_val.flow_from_directory(directory=val_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)
generator_test=datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                 batch_size=batch_size,
                                                 shuffle=False)

headModel = model1.output
headModel = layers.GlobalAveragePooling2D()(headModel)
headModel = Dropout(0.6)(headModel)
headModel = Dense(512, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(headModel)
headModel = layers.BatchNormalization()(headModel)
headModel = Dense(256, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(headModel)
headModel = layers.BatchNormalization()(headModel)
headModel = Dense(64, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(headModel)
headModel = Dense(2, activation="sigmoid")(headModel)
modelA = Model(inputs=model1.input, outputs=headModel)



headModel = model2.output
headModel = layers.GlobalAveragePooling2D()(headModel)
headModel = Dropout(0.6)(headModel)
headModel = Dense(512, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(headModel)
headModel = layers.BatchNormalization()(headModel)
headModel = Dense(256, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(headModel)
headModel = layers.BatchNormalization()(headModel)
headModel = Dense(64, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(headModel)
headModel = Dense(2, activation="sigmoid")(headModel)
modelB = Model(inputs=model2.input, outputs=headModel)



headModel = model3.output
headModel = layers.GlobalAveragePooling2D()(headModel)
headModel = Dropout(0.6)(headModel)
headModel = Dense(512, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(headModel)
headModel = layers.BatchNormalization()(headModel)
headModel = Dense(256, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(headModel)
headModel = layers.BatchNormalization()(headModel)
headModel = Dense(64, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(headModel)
headModel = Dense(2, activation="sigmoid")(headModel)
modelC = Model(inputs=model3.input, outputs=headModel)


import tensorflow as tf
models = [modelA,modelB,modelC]
model_input = tf.keras.Input(shape=(299, 299, 3))
model_outputs = [model(model_input) for model in models]
ensemble_output = tf.keras.layers.Average()(model_outputs)
ensemble_model = tf.keras.models.Model(inputs=model_input, outputs=ensemble_output, name='ensemble')

from keras import backend as K
def Recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def Precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def Specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


optimizer = Adam(learning_rate=1e-5)
loss = 'categorical_crossentropy'
metrics = ['accuracy', Precision, Recall, F1, Specificity]

ensemble_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

tf.keras.utils.plot_model(ensemble_model, 'model.png', show_shapes= True)


steps_per_epoch = generator_train.n / batch_size
steps_test = generator_test.n / batch_size

earlystopping = EarlyStopping(monitor ="val_loss",
                              mode ="min", patience = 5,
                              restore_best_weights = True)

history = ensemble_model.fit_generator(generator=generator_train,
                              epochs=100,
                              validation_data=generator_val,
                              validation_steps=generator_val.n / batch_size,
                              callbacks =[earlystopping]
                              )


import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
Y_pred = ensemble_model.predict_generator(generator_test)
y_pred = np.argmax(Y_pred, axis=1)
array = confusion_matrix(generator_test.classes, y_pred)
df_cm = pd.DataFrame(array, index =['Healthy Betel Leaf','Unhealthy Betel Leaf'],
                  columns = ['Healthy Betel Leaf','Unhealthy Betel Leaf'])
plt.figure(figsize=(10,10))
sn.set(font_scale=2)
sn.heatmap(df_cm, annot=True, cmap='summer', fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('model1_confusion',dpi=200);
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(20, 5))
ax = ax.ravel()

for i, met in enumerate(['Precision', 'Recall']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].grid(color = '#e0e0eb')
    ax[i].legend(['train', 'val'])

fig.savefig('model1_Precision_Recall_',dpi=1200);


fig, ax = plt.subplots(1, 2, figsize=(20, 5))
ax = ax.ravel()
for i, met in enumerate([ 'accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].grid(color = '#e0e0eb')
    ax[i].legend(['train', 'val'],loc='best')

fig.savefig('model1_accuracy_loss',dpi=1200);

plt.show()


fig, ax = plt.subplots(1, 2, figsize=(20, 5))
ax = ax.ravel()
for i, met in enumerate([ 'F1','Specificity']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].grid(color = '#e0e0eb')
    ax[i].legend(['train', 'val'],loc='best')

fig.savefig('model1_F1_Specificity',dpi=1200);

plt.show()

test_loss, test_accuracy, test_precision, test_recall, test_f1, test_specificity = ensemble_model.evaluate(generator_test, steps=steps_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1: {test_f1}")
print(f"Test Specificity: {test_specificity}")

Y_pred = ensemble_model.predict(generator_test, steps=steps_test)

auroc = roc_auc_score(generator_test.classes, Y_pred[:,1])
print(f"Test AUROC: {auroc}")

