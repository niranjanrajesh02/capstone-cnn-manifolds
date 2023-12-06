import tensorflow as tf
import numpy as np


def get_data():
  print("Getting Data ...")

  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

  print("Training Images Shape: ", train_images.shape)
  print("Training Labels Shape: ", train_labels.shape)
  print("Testing Images Shape: ", test_images.shape)
  print("Testing Labels Shape: ", test_labels.shape)

  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

  # Converting the pixels data to float type
  train_images = train_images.astype('float32')
  test_images = test_images.astype('float32')
  
  # Standardizing (255 is the total number of pixels an image can have)
  train_images = train_images / 255
  test_images = test_images / 255 

  # One hot encoding the target class (labels)
  num_classes = 10
  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

  return (train_images, train_labels), (test_images, test_labels)