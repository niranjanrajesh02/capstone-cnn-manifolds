from keras.applications.xception import Xception, preprocess_input, decode_predictions
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import os


model = Xception(weights='imagenet')

data_path = "../vid2img/frames/sunglass"
os.listdir(data_path)

