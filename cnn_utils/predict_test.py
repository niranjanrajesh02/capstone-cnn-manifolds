import tensorflow as tf
from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import argparse

def predict_test(model_name, image_path):
  if model_name == 'xception':
    from keras.applications.xception import Xception, preprocess_input, decode_predictions
    model = Xception(weights='imagenet')
  elif model_name == 'vgg16':
    from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    model = VGG16(weights='imagenet')
  elif model_name == 'resnet50':
    from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
    model = ResNet50(weights='imagenet')


  img = image.load_img(image_path)
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = preprocess_input(img_array)

  predictions = model.predict(img_array)

  decoded_predictions = decode_predictions(predictions, top=3)[0]
  for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
      print(f"{i + 1}: {label} ({score:.2f})")
  return decoded_predictions

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='xception', help='model name')
  parser.add_argument('--image_path', type=str, default='../vid2img/frames/mug_test/frame_0.jpg', help='image path')
  args = parser.parse_args()
  predict_test(args.model_name, args.image_path)