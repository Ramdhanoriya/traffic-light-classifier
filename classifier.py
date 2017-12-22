import cv2 # computer vision library
import helpers # helper functions
import features # functions to determine features

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images


def create_feature(rgb_image):

  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV) # Convert to HSV color space

  sum_brightness = np.sum(hsv[:,:,2]) # Sum the brightness values
  area = 32*32
  avg_brightness = sum_brightness / area # Find the average

  return avg_brightness

def standardize_input(image):

  # Shrink all images to be 32x32 px
  standard_im = cv2.resize(image, (32,32))
  return standard_im

def one_hot_encode(label):

  # Return the correct encoded label. A bit brute force, but it works.
  if label == 'red':
      return [1, 0, 0]
  if label == 'yellow':
      return [0, 1, 0]
  return [0, 0, 1]

def standardize(image_list):

  # Empty image data array
  standard_list = []

  # Iterate through all the image-label pairs
  for item in image_list:
    image = item[0]
    label = item[1]

    # Standardize the image
    standardized_im = standardize_input(image)

    # One-hot encode the label
    one_hot_label = one_hot_encode(label)

    # Append the image, and it's one hot encoded label to the full, processed list of image data
    standard_list.append((standardized_im, one_hot_label))

  return standard_list

def get_misclassified_images(test_images):
  # Track misclassified images by placing them into a list
  misclassified_images_labels = []

  # Iterate through all the test images
  # Classify each image and compare to the true label
  for image in test_images:

    # Get true data
    im = image[0]
    true_label = image[1]
    assert(len(true_label) == 3), "The true_label is not the expected length (3)."

    # Get predicted label from your classifier
    predicted_label = features.estimate_label(im)
    assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

    # Compare true and predicted labels
    if(predicted_label != true_label):
      # If these labels are not equal, the image has been misclassified
      misclassified_images_labels.append((im, predicted_label, true_label))

  # Return the list of misclassified [image, predicted_label, true_label] values
  return misclassified_images_labels

if __name__ == '__main__':
  # Using the load_dataset function in helpers.py
  # Load test data

  IMAGE_DIR_TRAINING = "traffic_light_images/training/"
  IMAGE_DIR_TEST = "traffic_light_images/test/"

  TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

  # Standardize the test data
  STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

  # Standardize all training images
  # STANDARDIZED_LIST = standardize(IMAGE_LIST)

  # Shuffle the standardized test data
  random.shuffle(STANDARDIZED_TEST_LIST)

  # Find all misclassified images in a given test set
  MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

  # Accuracy calculations
  total = len(STANDARDIZED_TEST_LIST)
  num_correct = total - len(MISCLASSIFIED)
  accuracy = num_correct/total

  print('Accuracy: ' + str(accuracy))
  print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

  random.shuffle(MISCLASSIFIED)
  f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))

  ax1.imshow(MISCLASSIFIED[0][0])
  ax2.imshow(MISCLASSIFIED[1][0])
  ax3.imshow(MISCLASSIFIED[2][0])
  ax4.imshow(MISCLASSIFIED[3][0])

  print(features.avg_red(MISCLASSIFIED[0][0]))
  print(features.avg_green(MISCLASSIFIED[0][0]))
  plt.show()
