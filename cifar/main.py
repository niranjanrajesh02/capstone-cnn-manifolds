from data import get_data
from nn import build_model
import numpy as np    
import matplotlib.pyplot as plt 
from keras.callbacks import EarlyStopping
import pandas as pd

(train_images, train_labels), (test_images, test_labels) = get_data()

cnn_model = build_model(num_classes=10)

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# early stop
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)

# train model
history = cnn_model.fit(train_images, train_labels, epochs=1, batch_size=64, validation_data=(test_images, test_labels), callbacks=[early_stop])

# save history as csv with df
df = pd.DataFrame(history.history)
df.to_csv('./history.csv')

# save model
cnn_model.save('./trained_cifar10_model.h5')


# tfversion

