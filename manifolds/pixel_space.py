import cv2
import numpy as np
import tensorflow as tf
import keras
import os
import pandas as pd


def get_flattened_imgs(img_size=299, env='pc'):
  print("Flattening images...")

  


  if env == 'hpc':
    n_dim = img_size * img_size * 3
    frames_path = "/home/niranjan.rajesh_asp24/capstone-cnn-manifolds/vid2img/frames/"
  elif env == 'pc':
    n_dim = 100
    frames_path = "../vid2img/frames/"

  classes = os.listdir(frames_path)
  class_images = {}

  for class_i in classes:
      class_path = os.path.join(frames_path, class_i)
      class_len = len(os.listdir(class_path))
      # class_images[class_i] = np.zeros((class_len, img_size*img_size))
      class_images[class_i] = np.zeros((class_len, n_dim))
      # store an np array of all the flattened images in the class
      for i, img in enumerate(os.listdir(class_path)):
          img_path = os.path.join(class_path, img)
          img = cv2.imread(img_path, 0)
          img = img.flatten()[:100]
          class_images[class_i][i] = img
          print(f"Flattened image {i} in class {class_i}")
          
  
  return class_images




