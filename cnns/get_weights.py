import tensorflow as tf
from keras.applications.xception import Xception, preprocess_input, decode_predictions

model = Xception(weights='imagenet')
model.save_weights('./weights/xception_imagenet_weights.h5')

model = tf.keras.applications.ResNet50(weights='imagenet')
model.save_weights('./weights/resnet50_imagenet_weights.h5')

model = tf.keras.applications.VGG16(weights='imagenet')
model.save_weights('./weights/vgg16_imagenet_weights.h5')