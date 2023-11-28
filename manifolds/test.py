import numpy as np

# Assuming arr is your array of shape (5, 30, 10, 10)
arr = np.random.random((5, 30, 10, 10))

# Reshape the array
new_shape = (-1, arr.shape[-2], arr.shape[-1])
reshaped_arr = arr.reshape(new_shape)

# Display the shapes
print("Original shape:", arr.shape)
print("Reshaped shape:", reshaped_arr.shape)