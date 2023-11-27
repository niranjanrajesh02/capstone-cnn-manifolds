from pixel_space import get_flattened_imgs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import cv2
from mani_utils import load_data_from_dict
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--space', type=str, default='pixel', help='Space you want to classify')  # sunglass or mug
parser.add_argument('--env', type=str, default='pc', help='environment')
args = parser.parse_args()


def classify_pixel_space():
  class_imgs = get_flattened_imgs(env=args.env)



  # create X and Y for linear model training
  X, Y = load_data_from_dict(class_imgs)
  del class_imgs
  # Train linear model
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
  model = LogisticRegression()
  model.fit(X_train, Y_train)
  Y_pred = model.predict(X_test)
  print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")


  # create a dataframe of the classes

  return


classify_pixel_space()