import tensorflow as tf
from data import get_binary_cifar_data
import numpy as np
from layer_activations import getLayerActivations, layer_names
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import time
import pandas as pd
from sklearn.linear_model import SGDClassifier
import argparse

def check_separability(c1, c2, clf="logreg"):
    
  # get binary data
  (train_images, train_labels), (test_images, test_labels) = get_binary_cifar_data('pc', c1, c2)

  print("Getting Layer Activations ...")

  layers = list(layer_names.keys())

  # save df initialise
  save_df = pd.DataFrame(index=np.arange(len(layers)+1), columns=["layer_name", "class1", "class2", "acts_dim", "train_accuracy", "test_accuracy", "train_time"])
  index = 0

  for layer in layers:

    print("\n=============================================")
    print(f"Checking Linear Separability for {layer} ... \n")
    # get layer activations from cnn
    train_acts = getLayerActivations(layer, train_images)
    test_acts = getLayerActivations(layer, test_images)

    # flatten activations
    train_acts_flat = train_acts.reshape(train_acts.shape[0], -1)
    test_acts_flat = test_acts.reshape(test_acts.shape[0], -1)

    # scale activations
    scaler = preprocessing.StandardScaler()
    train_acts_flat = scaler.fit_transform(train_acts_flat)
    test_acts_flat = scaler.transform(test_acts_flat)
    print("train_acts_flat shape: ", train_acts_flat.shape)
    print("test_acts_flat shape: ", test_acts_flat.shape)

    if clf == "logreg":
      # train logistic regression
      lmodel = LogisticRegression()
    elif lmodel == "svm":
      # train svm
      lmodel = LinearSVC()
    elif clf == "sgd":
      # train sgd
      lmodel = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
    else:
      print("Invalid Classifier")
      return
    

    time1 = time.time()
    lmodel.fit(train_acts_flat, np.argmax(train_labels, axis=1))
    time2 = time.time()
    print("Training Complete\nElapsed Time: ", time2-time1, " seconds\n\nChecking Separability ...")
    # test logistic regression
    train_preds = lmodel.predict(train_acts_flat)
    test_preds = lmodel.predict(test_acts_flat)
    # get accuracy
    train_acc = accuracy_score(np.argmax(train_labels, axis=1), train_preds)
    test_acc = accuracy_score(np.argmax(test_labels, axis=1), test_preds)
    print("Train Accuracy: ", train_acc)
    print("Test Accuracy: ", test_acc)

    # save stats to save_df
    save_df.loc[index] = [layer, c1, c2, train_acts_flat.shape[1], train_acc, test_acc, time2-time1]
    index += 1
    print("=============================================")

  return save_df

def check_pixel_separability(c1, c2, clf="logreg"):
  # get binary data
  (train_images, train_labels), (test_images, test_labels) = get_binary_cifar_data('pc', c1, c2)

  # flatten images
  train_images_flat = train_images.reshape(train_images.shape[0], -1)
  test_images_flat = test_images.reshape(test_images.shape[0], -1)

  # scale images
  scaler = preprocessing.StandardScaler()
  train_images_flat = scaler.fit_transform(train_images_flat)
  test_images_flat = scaler.transform(test_images_flat)

  if clf == "logreg":
    # train logistic regression
    lmodel = LogisticRegression()
  elif lmodel == "svm":
    # train svm
    lmodel = LinearSVC()
  elif clf == "sgd":
    # train sgd
    lmodel = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
  else:
    print("Invalid Classifier")
    return
  
  time1 = time.time()
  lmodel.fit(train_images_flat, np.argmax(train_labels, axis=1))
  time2 = time.time()
  print("Training Complete\nElapsed Time: ", time2-time1, " seconds\n\nChecking Separability ...")
  # test logistic regression
  train_preds = lmodel.predict(train_images_flat)
  test_preds = lmodel.predict(test_images_flat)
  # get accuracy
  train_acc = accuracy_score(np.argmax(train_labels, axis=1), train_preds)
  test_acc = accuracy_score(np.argmax(test_labels, axis=1), test_preds)
  print("Train Accuracy: ", train_acc)
  print("Test Accuracy: ", test_acc)

  

  return train_acc, test_acc, time2-time1

def test_classifers(train_acts, test_acts, train_labels, test_labels):
  # flatten activations
  train_acts_flat = train_acts.reshape(train_acts.shape[0], -1)
  test_acts_flat = test_acts.reshape(test_acts.shape[0], -1)

  # train sgd classifier
  time1 = time.time()

  sgd = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
  sgd.fit(train_acts_flat, np.argmax(train_labels, axis=1))

  time2 = time.time()

  print("Time taken for SGD: ", time2-time1)

  print("Training Complete")
  # test sgd
  train_preds = sgd.predict(train_acts_flat)
  test_preds = sgd.predict(test_acts_flat)

  # get accuracy
  train_acc = accuracy_score(np.argmax(train_labels, axis=1), train_preds)
  test_acc = accuracy_score(np.argmax(test_labels, axis=1), test_preds)

  print("Train Accuracy: ", train_acc)
  print("Test Accuracy: ", test_acc)


  time1 = time.time()

  # train svm
  svm = LinearSVC()
  svm.fit(train_acts_flat, np.argmax(train_labels, axis=1))

  time2 = time.time()

  print("Time taken for SVM: ", time2-time1)

  print("Training Complete")
  # test svm
  train_preds = svm.predict(train_acts_flat)
  test_preds = svm.predict(test_acts_flat)

  # get accuracy
  train_acc = accuracy_score(np.argmax(train_labels, axis=1), train_preds)
  test_acc = accuracy_score(np.argmax(test_labels, axis=1), test_preds)

  print("Train Accuracy: ", train_acc)
  print("Test Accuracy: ", test_acc)



  print("=============================================")

  time1 = time.time()

  # train logistic regression
  logreg = LogisticRegression()
  logreg.fit(train_acts_flat, np.argmax(train_labels, axis=1))

  time2 = time.time()

  print("Time taken for Logistic Regression: ", time2-time1)

  print("Training Complete")

  # test logistic regression
  train_preds = logreg.predict(train_acts_flat)
  test_preds = logreg.predict(test_acts_flat)

  # get accuracy
  train_acc = accuracy_score(np.argmax(train_labels, axis=1), train_preds)
  test_acc = accuracy_score(np.argmax(test_labels, axis=1), test_preds)

  print("Train Accuracy: ", train_acc)
  print("Test Accuracy: ", test_acc)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--class1', type=str, default='truck', help='class1')
  parser.add_argument('--class2', type=str, default='automobile', help='class2')
  args = parser.parse_args()

  save_df = check_separability(args.class1, args.class2)
  train_acc, test_acc, time_taken =  check_pixel_separability(args.class1, args.class2)
  save_df.loc[len(save_df)-1] = ["pixel", args.class1, args.class2, (32*32*3), train_acc, test_acc, time_taken]
  save_df.to_csv("./reps_dicts/"+args.class1+"_"+args.class2+"_reps.csv")
  print("All Done! xD")
  

