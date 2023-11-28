import tensorflow as tf
from mani_utils import load_model_layernames
from keras.preprocessing import image
import os
import cv2
import numpy as np
import gc
# function that returns the activations of a specific layer given an imagenet pretrained model when passed an image

def iterateImgs(imgdir_path, model, preproc, layer_name, env='pc', every_n_img=1):
  classes = os.listdir(imgdir_path)
  class_reps = {}

  for class_i in classes:
      class_path = os.path.join(imgdir_path, class_i)
      class_len = len(os.listdir(class_path)) // every_n_img + 1
      layer = model.get_layer(layer_name)
      if env == 'hpc':
        n_dim = layer.output.shape[1] * layer.output.shape[2] * layer.output.shape[3]
      else:
        n_dim = 100
      print("Representational Space Dimension: ", n_dim)
      class_reps[class_i] = np.zeros((class_len, n_dim), dtype=int)
      # store an np array of all the flattened images in the class
      count = 0
      for i, img in enumerate(os.listdir(class_path)):
          if i % every_n_img != 0:
            continue
          img_path = os.path.join(class_path, img)
          img = cv2.imread(img_path).astype('int')
          # predict the image
          img = tf.expand_dims(img, axis=0)
          img /= 255
          reps = model.predict(img)
          print("Predicted for image ", i, " in class ", class_i, " of ", class_len, " images")
          reps = reps.flatten() 
          class_reps[class_i][count] = reps[:n_dim]
          count += 1
      print(f"Got representations for class {class_i}")    
  return class_reps



def getRepresentations(model_name, layer_ind, env='pc', every_n=1):
  if env == 'hpc':
    root_path = "/home/niranjan.rajesh_asp24/capstone-cnn-manifolds/"
    img_path = "/home/niranjan.rajesh_asp24/capstone-cnn-manifolds/vid2img/frames/"
    weights_path = f"/home/niranjan.rajesh_asp24/capstone-cnn-manifolds/cnns/weights/{model_name}_imagenet_weights.h5"
  elif env == 'pc':
    root_path = "../"
    img_path = "../vid2img/frames/"
    weights_path = f"../cnns/weights/{model_name}_imagenet_weights.h5"

  if model_name == 'xception':
    from keras.applications.xception import Xception, preprocess_input, decode_predictions
    model = Xception(weights=None)
    model.load_weights(weights_path)
    layers_df = load_model_layernames('xception', root_path)
    layer_name = layers_df.iloc[layer_ind]['layer_name']
    del layers_df
    gc.collect()
    model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    print("Model loaded successfully")
    print("Getting activations ...")
    reps_dict = iterateImgs(imgdir_path=img_path, model=model, preproc=preprocess_input, layer_name=layer_name , env=env, every_n_img=every_n)
    del model
    gc.collect()
    return reps_dict


# getRepresentations(model_name='xception', layer_ind=0, env='pc', every_n=100)