from data import get_data
from nn import build_model
import numpy as np    
import matplotlib.pyplot as plt 
from keras.callbacks import EarlyStopping
import pandas as pd
import argparse

# args
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='hpc', help='hpc or pc')
args = parser.parse_args()


train_ds, test_ds = get_data(env=args.env)

cnn_model = build_model(num_classes=10)

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# early stop
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)


# train model
if args.env == 'pc':
  history = cnn_model.fit(train_ds[0], train_ds[1], epochs=1, batch_size=64, validation_data=(test_ds[0], test_ds[1]), callbacks=[early_stop])
else:
  history = cnn_model.fit(train_ds, epochs=1, batch_size=64, validation_data=test_ds, callbacks=[early_stop])

# save history as csv with df
df = pd.DataFrame(history.history)

if args.env == 'pc':
  df.to_csv('./cifar_mobnet_history.csv')
  cnn_model.save('trained_cifar_mobnet.h5')
else:
  store_path = '/home/niranjan.rajesh_asp24/capstone-cnn-manifolds/Results'
  df.to_csv(store_path+'cifar_mobnet_history.csv')
  cnn_model.save(store_path+'trained_cifar10_model.h5')


