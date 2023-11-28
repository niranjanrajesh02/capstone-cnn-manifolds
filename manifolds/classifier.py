from pixel_space import get_flattened_imgs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import cv2
from mani_utils import load_data_from_dict
import argparse 
from representation_space import getRepresentations

parser = argparse.ArgumentParser()
parser.add_argument('--space', type=str, default='pixel', help='Space you want to classify')  
parser.add_argument('--env', type=str, default='pc', help='environment')
parser.add_argument('--layer', type=int, default=0, help='layer number', required=False)
parser.add_argument('--model', type=str, default='xception', help='model name', required=False)

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


def classify_layer_space(layer_num):
  class_reps = getRepresentations(args.model, args.env, args.layer)
  # create X and Y for linear model training
  X, Y = load_data_from_dict(class_reps)
  del class_reps
  # Train linear model
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=42)
  model = LogisticRegression()
  model.fit(X_train, Y_train)
  Y_pred = model.predict(X_train)
  print(f"Accuracy: {accuracy_score(Y_train, Y_pred)}")
  return

if __name__ == '__main__':
  if args.space == 'pixel':
    classify_pixel_space()
  elif args.space == 'layer':
    classify_layer_space(0)