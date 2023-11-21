import cv2
import numpy as np


def center_crop(img, dim):
  """Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
  width, height = img.shape[1], img.shape[0]
	# process crop width and height for max available dimension
  crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
  crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
  mid_x, mid_y = int(width/2), int(height/2)
  cw2, ch2 = int(crop_width/2), int(crop_height/2) 
  crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
  return crop_img



def central_resize(img, output_size=(224, 224)):
    # Get the height and width of the original image
    original_height, original_width = img.shape[:2]

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_width / original_height

    # Calculate the new height and width to maintain the aspect ratio
    new_width = int(min(output_size[0], output_size[1] * aspect_ratio))
    new_height = int(min(output_size[1], output_size[0] / aspect_ratio))

    # Resize the image while maintaining the aspect ratio
    resized_img = cv2.resize(img, (new_width, new_height))

    # Create a blank white image with the desired output size
    result_img = 255 * np.ones((output_size[1], output_size[0], 3), dtype=np.uint8)

    # Calculate the position to paste the resized image in the center
    x_offset = (output_size[0] - new_width) // 2
    y_offset = (output_size[1] - new_height) // 2

    # Paste the resized image onto the blank white image
    result_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

    return result_img