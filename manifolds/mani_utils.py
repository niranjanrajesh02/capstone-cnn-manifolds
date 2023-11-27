import numpy as np

# creates X and Y for linear model training from a dictionary of form {label: [x1, x2, ...], label2: [x'1, x'2, ...]}
def load_data_from_dict(data):
    X = []
    Y = []
    
    for y, x_list in data.items():
        for x in x_list:
            X.append(x)  
            Y.append(y)
    
    return np.array(X), np.array(Y)