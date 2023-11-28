import tensorflow as tf
from keras.applications.xception import Xception, preprocess_input, decode_predictions

model = Xception(weights='imagenet')
model.save_weights('./weights/xception_imagenet_weights.h5')
