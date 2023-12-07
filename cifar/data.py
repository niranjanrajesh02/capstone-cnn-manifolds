import tensorflow as tf
import numpy as np


def get_data(env):
  print("Getting Data ...")
  if env == 'pc':
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

  else:
    data_path = '/storage/niranjan.rajesh_asp24/cifar10/cifar10'
    # load tf dataset from directory
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_path+'/train',
      labels='inferred',
      label_mode='categorical',
      class_names=None,
      color_mode='rgb',
      batch_size=64,
      image_size=(32, 32),
      shuffle=True,
      seed=None,
      validation_split=None,
      subset=None)
    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_path+'/test',
      labels='inferred',
      label_mode='categorical',
      class_names=None,
      color_mode='rgb',
      batch_size=64,
      image_size=(32, 32),
      shuffle=True,
      seed=None )
    
    return train_ds, valid_ds
  

cifar_classes = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,
                'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}


def get_class_cifar_data(env, class_name):
  class_idx = cifar_classes[class_name]
  print("Getting Data ...")
  # get_data call
  (train_images, train_labels), (test_images, test_labels) = get_data(env)
  # get class indices
  train_class_indices = np.where(train_labels[:,class_idx]==1)[0]
  test_class_indices = np.where(test_labels[:,class_idx]==1)[0]
  # get class data
  train_class_images = train_images[train_class_indices]
  test_class_images = test_images[test_class_indices]
  train_class_labels = train_labels[train_class_indices]
  test_class_labels = test_labels[test_class_indices]
  
  return (train_class_images, train_class_labels), (test_class_images, test_class_labels)

def get_binary_cifar_data(env, class1, class2):
  
  # get horse data
  c1_train, c1_test = get_class_cifar_data('pc', class1)

  # get bird data
  c2_train, c2_test = get_class_cifar_data('pc', class2)

  # stack horse and bird data
  train_images = np.vstack((c1_train[0], c2_train[0]))
  train_labels = np.vstack((c1_train[1], c2_train[1]))
  test_images = np.vstack((c1_test[0], c2_test[0]))
  test_labels = np.vstack((c1_test[1], c2_test[1]))

  return (train_images, train_labels), (test_images, test_labels)