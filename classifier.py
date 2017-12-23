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
  IMAGE_DIR_TRAINING = "traffic_light_images/training/"
  IMAGE_DIR_TEST = "traffic_light_images/test/"

  TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

  # Standardize the test data
  STANDARDIZED_TEST_LIST = helpers.standardize(TEST_IMAGE_LIST)

  # Shuffle the standardized test data
  random.shuffle(STANDARDIZED_TEST_LIST)
  image = STANDARDIZED_TEST_LIST[0][0]

  # Find all misclassified images in a given test set
  MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

  # Accuracy calculations
  total = len(STANDARDIZED_TEST_LIST)
  num_correct = total - len(MISCLASSIFIED)
  accuracy = num_correct/total

  print('Accuracy: ' + str(accuracy))
  print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

  # red_missclassified = []
  # for image in MISCLASSIFIED:
  #   if image[2] == [1, 0, 0]:
  #     red_missclassified.append(image)
  # print(len(red_missclassified))

  # image = red_missclassified[0][0]
  r = image[:,:,0]
  g = image[:,:,1]
  b = image[:,:,2]

  hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  h = hsv[:,:,0]
  s = hsv[:,:,1]
  v = hsv[:,:,2]

  # random.shuffle(red_missclassified)
  f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))

  ax1.imshow(image)
  ax2.imshow(h)
  ax3.imshow(s)
  ax4.imshow(v)

  print(features.high_saturation_pixels(image)[0])
  # print(features.avg_red(red_missclassified[0][0]))
  # print(features.avg_green(red_missclassified[0][0]))
  plt.show()
