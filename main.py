import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from utils import download_dataset, preprocess, plot_accuracy,plot_loss, plot_confusion_matrix

os.system('cls')

print('\n-----Downloading dataset------ \n')
download_dataset()

# Paths to train and test 
train_dir = './1/train'
test_dir = './1/test' 

print("\n")

# Setting up parameters
img_size = 48  # FER2013 images are 48x48
batch_size = 32
num_classes = 7
learning_rate = 0.0001
epochs = 25
weights_path = "vgg_facial_expression.keras"


# Load the train and test dataset
train_dataset = image_dataset_from_directory(
    Path(train_dir),
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical"  # one-hot labels
)

validation_dataset = image_dataset_from_directory(
    test_dir,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical" 
)

# Data augmentation for training dataset
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2)
])

# Save class labels
class_names = train_dataset.class_names

train_dataset = train_dataset.map(preprocess).map(lambda x, y: (data_augmentation(x), y))
validation_dataset = validation_dataset.map(preprocess)

# Prefetch for performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Load VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

# Add custom layers for FER2013 classification
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)  # Added L2 regularization
x = Dropout(0.5)(x)
out = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=out)
print('\n-----Details of the model------ \n')
model.summary()


# Check if the weights file exists
if os.path.exists(weights_path):
    print(f"\n ----------Loading pre-trained weights from {weights_path}-----------\n")
    model.load_weights(weights_path)
    
     # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate / 10),  # Use a smaller learning rate for consistency
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Evaluate model on the test data
    print("\n-----------Testing the model with pre-trained weights-----------\n")
    test_loss, test_accuracy = model.evaluate(validation_dataset)
    print(f"        Test Accuracy: {test_accuracy:.4f}")
    
else:
    print("\n----------No pre-trained weights found. Training the model...-------------\n")
 
    # Freeze the base VGG16 layers initially
    for layer in base_model.layers:
        layer.trainable = False
        
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    checkpoint = ModelCheckpoint("vgg_facial_expression.keras", monitor="val_accuracy", save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )
    
     print("\n----------Unfreezing VGG16 layers. Fine-tuning the model...-------------\n")
    # Fine-tuning: Unfreeze all layers
    for layer in base_model.layers:
        layer.trainable = True

    # Recompile the model with a smaller learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=learning_rate / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Fine-tune training
    history_finetune = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )

# Ploting results
    print("\n----------Ploting results-------------\n")
    plot_accuracy(history, title="Initial Training Accuracy")
    plot_loss(history, title="Initial Training Loss")
    plot_accuracy(history_finetune, title="Fine-Tuning Accuracy")
    plot_loss(history_finetune, title="Fine-Tuning Loss")
    
plot_confusion_matrix(model, validation_dataset, class_names)