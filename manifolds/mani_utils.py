import numpy as np
import pandas as pd
# creates X and Y for linear model training from a dictionary of form {label: [x1, x2, ...], label2: [x'1, x'2, ...]}
def load_data_from_dict(data):
    X = []
    Y = []
    
    for y, x_list in data.items():
        for x in x_list:
            X.append(x)  
            Y.append(y)
    
    return np.array(X), np.array(Y)

def write_model_layernames(model_name):
    if model_name == 'xception':
        from keras.applications.xception import Xception, preprocess_input, decode_predictions
        model = Xception(weights='imagenet')
        layer_names = [layer.name for layer in model.layers if layer.name.endswith('act')]
        del model
        # save layer names to csv file
        np.savetxt("./xception_layer_names.csv", layer_names, delimiter=",", fmt='%s')

def load_model_layernames(model_name):
    if model_name == 'xception':
        # load layer names from csv file as dataframe with index
        layer_names = pd.read_csv('./xception_layer_names.csv', header=None, names=['layer_name'])
        return layer_names