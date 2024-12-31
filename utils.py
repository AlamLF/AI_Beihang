import kagglehub
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Downloads the FER2013 dataset
def download_dataset():
    
    path = kagglehub.dataset_download("msambare/fer2013")
    target_directory = os.getcwd()
    os.makedirs(target_directory, exist_ok=True)
    shutil.move(path, target_directory)

    print("Dataset moved to:", target_directory)
    
    return(target_directory)

# Normalize pixel values
def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

# Plot accuracy
def plot_accuracy(history, title="Model Accuracy"):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

# Plot loss
def plot_loss(history, title="Model Loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


# Confusion Matrix
def plot_confusion_matrix(model, dataset, class_names):
    y_pred = []
    y_true = []

    for images, labels in dataset:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()