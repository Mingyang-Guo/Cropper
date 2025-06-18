import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix , classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

# Remove the wandb import and related code
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# wandb_api = user_secrets.get_secret("wandb_api")
# wandb.login(key=wandb_api)

train_dir = "/home/kaga/Desktop/mingyang/dataset/Unique2/RiceSmall/train80"
val_dir = "/home/kaga/Desktop/mingyang/dataset/Unique2/RiceSmall/validation20"
test_dir = "/home/kaga/Desktop/mingyang/dataset/Unique2/RiceSmall/Test"

SEED = np.random.randint(1, 10000)

print(f"Random Seed: {SEED}")

a = SEED

IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE =8
EPOCHS = 100
#FINE_TUNING_EPOCHS = 20
LR = 0.01
CLASS_LABELS  = ['bacterial_leaf_blight','brown_spot','healthy','leaf_blast','leaf_scald','narrow_brown_spot']
NUM_CLASSES = len(CLASS_LABELS)
EARLY_STOPPING_CRITERIA=5

CONFIG = dict (
    SEED = a,
    IMG_HEIGHT = 512,
    IMG_WIDTH = 512,
    BATCH_SIZE =8,
    EPOCHS = 100,
    #FINE_TUNING_EPOCHS = 20,
    LR = 0.01,
    CLASS_LABELS  =  ['bacterial_leaf_blight','brown_spot','healthy','leaf_blast','leaf_scald','narrow_brown_spot'],
    NUM_CLASSES = len(CLASS_LABELS),
    EARLY_STOPPING_CRITERIA=5,
)


train_datagen = ImageDataGenerator(
                                   rescale = 1./255,
                                  )
valid_datagen = ImageDataGenerator(rescale = 1./255,
                                 )
test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode="rgb",
    class_mode="categorical",
    seed=SEED
)

valid_generator = valid_datagen.flow_from_directory(directory=val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="rgb",
    class_mode="categorical",
    seed=SEED
    )

test_generator = test_datagen.flow_from_directory(directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="rgb",
    class_mode="categorical",
    seed=SEED
    )


def display_one_image(image, title, subplot, color):
    plt.subplot(subplot)
    plt.axis('off')
    plt.imshow(image)
    plt.title(title, fontsize=16)


def display_nine_images(images, titles, title_colors=None):
    subplot = 331
    plt.figure(figsize=(13, 13))
    for i in range(4):
        color = 'black' if title_colors is None else title_colors[i]
        display_one_image(images[i], titles[i], 331 + i, color)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def image_title(label, prediction):
    # Both prediction (probabilities) and label (one-hot) are arrays with one item per class.
    class_idx = np.argmax(label, axis=-1)
    prediction_idx = np.argmax(prediction, axis=-1)
    if class_idx == prediction_idx:
        return f'{CLASS_LABELS[prediction_idx]} [correct]', 'black'
    else:
        return f'{CLASS_LABELS[prediction_idx]} [incorrect, should be {CLASS_LABELS[class_idx]}]', 'red'


def get_titles(images, labels, model):
    predictions = model.predict(images)
    titles, colors = [], []
    for label, prediction in zip(classes, predictions):
        title, color = image_title(label, prediction)
        titles.append(title)
        colors.append(color)
    return titles, colors


images, classes = next(train_generator)
class_idxs = np.argmax(classes, axis=-1)
labels = [CLASS_LABELS[idx] for idx in class_idxs]
display_nine_images(images, labels)


fig = px.bar(x = CLASS_LABELS,
             y = [list(train_generator.classes).count(i) for i in np.unique(train_generator.classes)] ,
             color = np.unique(train_generator.classes) ,
             color_continuous_scale="Emrld")
fig.update_xaxes(title="Fruit Images")
fig.update_yaxes(title="Number of Images")
fig.update_layout(
    showlegend=True,
    title={
        'text': 'Train Data Distribution ',
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
fig.show()

def feature_extractor(inputs):
    feature_extractor = tf.keras.applications.DenseNet169(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet")(inputs)
    return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation=tf.nn.silu, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)
    return x

def final_model(inputs):
    densenet_feature_extractor = feature_extractor(inputs)
    classification_output = classifier(densenet_feature_extractor)
    return classification_output

def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    classification_output = final_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=classification_output)
    model.compile(optimizer=tf.keras.optimizers.SGD(LR), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = define_compile_model()
clear_output()

# model.summary()

CONFIG['model_name'] = 'densenet169'
print('Training configuration: ', CONFIG)

earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_CRITERIA,
    verbose=1,
    restore_best_weights=True
)

history = model.fit(
    x=train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    callbacks=[earlyStoppingCallback]
)

# Save the training history to a CSV file
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

# Save the model locally
model.save('densenet169-baseline.h5')

# Evaluate the model
preds = model.predict(test_generator)
y_preds = np.argmax(preds, axis=1)
y_test = np.array(test_generator.labels)

# Compute and print confusion matrix and classification report
cm_data = confusion_matrix(y_test, y_preds)
cm = pd.DataFrame(cm_data, columns=CLASS_LABELS, index=CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize=(20, 10))
plt.title('Confusion Matrix', fontsize=20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')

print(classification_report(y_test, y_preds))

# Compute and print ROC AUC score
fig, c_ax = plt.subplots(1, 1, figsize=(15, 8))
def multiclass_roc_auc_score(y_test_onehot, y_pred, average="macro"):
    if len(y_test_onehot.shape) == 1:
        y_test_onehot = to_categorical(y_test_onehot, num_classes=NUM_CLASSES)

    n_classes = y_test_onehot.shape[1]
    for idx in range(n_classes):
        c_label = CLASS_LABELS[idx]
        fpr, tpr, _ = roc_curve(y_test_onehot[:, idx], y_pred[:, idx])
        plt.plot(fpr, tpr, lw=2, label=f'{c_label} (AUC:{auc(fpr, tpr):0.2f})')

    plt.plot([0,1], [0,1], 'black', linestyle='dashed', lw=4, label='Random Guessing')
    return roc_auc_score(y_test_onehot, y_pred, average=average)


from sklearn.metrics import (accuracy_score, precision_score,
                            recall_score, f1_score, roc_auc_score)


if len(y_test.shape) > 1:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels = y_test


test_accuracy = accuracy_score(y_test_labels, y_preds)
test_precision = precision_score(y_test_labels, y_preds, average='macro')
test_recall = recall_score(y_test_labels, y_preds, average='macro')
test_f1 = f1_score(y_test_labels, y_preds, average='macro')
test_auroc = multiclass_roc_auc_score(y_test, preds, average="micro")


cm_data = confusion_matrix(y_test, y_preds)
if NUM_CLASSES == 2:
    tn, fp, fn, tp = cm_data.ravel()
    specificity = tn / (tn + fp)
else:
    specificities = []
    for i in range(NUM_CLASSES):
        tn = np.sum(np.delete(np.delete(cm_data, i, axis=0), i, axis=1))
        fp = np.sum(cm_data[:, i]) - cm_data[i, i]
        specificities.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
    specificity = np.mean(specificities)


preds = model.predict(test_generator)
y_preds = np.argmax(preds, axis=1)
y_test = np.array(test_generator.labels)



y_test_onehot = to_categorical(y_test, num_classes=NUM_CLASSES)
test_auroc = multiclass_roc_auc_score(y_test_onehot, preds, average="micro")

loss, acc = model.evaluate(test_generator)
print("\n" + "="*50)
print("Final Test Performance Metrics:")
print(f"• Test Accuracy:    {test_accuracy:.4f}")
print(f"• Test Precision:   {test_precision:.4f} (macro avg)")
print(f"• Test Recall:      {test_recall:.4f} (macro avg)")
print(f"• Test F1 Score:    {test_f1:.4f} (macro avg)")
print(f"• Test Specificity: {specificity:.4f}")
if NUM_CLASSES == 2:
    print(f"  (Healthy specificity: {specificity:.4f})")
print(f"• Test AUROC:       {test_auroc:.4f} (micro avg)")
print("="*50 + "\n")

plt.xlabel('FALSE POSITIVE RATE', fontsize=18)
plt.ylabel('TRUE POSITIVE RATE', fontsize=16)
plt.legend(fontsize=6)
plt.show()

# Save the model and results locally (without W&B)
model.save('densenet169-baseline.h5')