import tensorflow as tf
from mani_utils import load_model_layernames
from keras.preprocessing import image
import os
import cv2
import numpy as np
# function that returns the activations of a specific layer given an imagenet pretrained model when passed an image

def iterateImgs(img_path, model, preproc, layer_name):
  classes = os.listdir(img_path)
  class_reps = {}

  for class_i in classes:
      class_path = os.path.join(img_path, class_i)
      class_len = len(os.listdir(class_path))
      layer = model.get_layer(layer_name)
      n_dim = layer.output.shape[1] * layer.output.shape[2] * layer.output.shape[3]
      print("Representational Space Dimension: ", n_dim)
      class_reps[class_i] = np.zeros((class_len, n_dim))
      # store an np array of all the flattened images in the class
      for i, img in enumerate(os.listdir(class_path)):
          img_path = os.path.join(class_path, img)
          img = cv2.imread(img_path)
          # predict the image
          img = tf.expand_dims(img, axis=0)
          img = preproc(img)
          reps = model.predict(img)
          reps = reps.flatten()
          class_reps[class_i][i] = reps
      
      return class_reps



def getRepresentations(model_name, layer_ind, env='pc'):
  if env == 'hpc':
    img_path = "/home/niranjan.rajesh_asp24/capstone-cnn-manifolds/vid2img/frames/"
  elif env == 'pc':
    img_path = "../vid2img/frames/"

  if model_name == 'xception':
    from keras.applications.xception import Xception, preprocess_input, decode_predictions
    model = Xception(weights='imagenet')
    print(model.summary())
    layers_df = load_model_layernames('xception')
    layer_name = layers_df.iloc[layer_ind]['layer_name']
    model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    reps_dict = iterateImgs(img_path=img_path, model=model, preproc=preprocess_input, layer_name=layer_name)
    print(reps_dict)
    print(reps_dict['mug'][0].shape)


getRepresentations(model_name='xception', layer_ind=0, env='pc')