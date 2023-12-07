import tensorflow as tf
import numpy as np

# load saved weights onto mobnet
cnn = tf.keras.applications.MobileNetV2(input_shape=(32,32,3), weights=None, classes=10)
cnn.load_weights('./weights/trained_cifar10_mobnet.h5')
print("Loaded Weights")


layer_names = {
  'Conv1': 1,
  'Conv_1': 151,
  'global_average_pooling2d': 154,
  'predictions': 155,
}

# checking layer indices
for layer_ind, layer in enumerate(cnn.layers):
  # print(layer_ind, layer.name, layer.output_shape)
  if ('project' in layer.name and 'BN' not in layer.name):
    layer_names[layer.name] = layer_ind


def getLayerActivations(layer_name, data):
  layer_index = layer_names[layer_name]
  print(f"Getting Layer Activations for {layer_index}th layer: {layer_name}")
  # get layer1 activations from cnn
  layer1 = cnn.layers[layer_index]
  layer_model = tf.keras.models.Model(inputs=cnn.inputs, outputs=layer1.output)
  activations = layer_model.predict(data)
  return activations