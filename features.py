import numpy as np

def create_feature(rgb_image):

  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV) # Convert to HSV color space

  sum_brightness = np.sum(hsv[:,:,2]) # Sum the brightness values
  area = 32*32
  avg_brightness = sum_brightness / area # Find the average

  return avg_brightness

def avg_red(rgb_image):
  # Determines the average red content in a standardized RGB image

  sum_red = np.sum(rgb_image[:,:,0]) # Sum the red values
  area = 32*32
  avg_red = sum_red / area # Find the average

  return avg_red

def max_red(rgb_image):
  return (np.max(rgb_image[:,:,0]))

def max_green(rgb_image):
  return (np.max(rgb_image[:,:,1]))

def avg_green(rgb_image):
  # Determines the average red content in a standardized RGB image

  sum_green = np.sum(rgb_image[:,:,1]) # Sum the red values
  area = 32*32
  avg_green = sum_green / area # Find the average

  return avg_green

def mask_saturation(rgb_image):
    # Returns average red and green content from high saturation pixels
  total_green = []
  total_red = []
  saturation_threshold = 100
  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
  # for pixel in hsv[:][:][1]:
    # if pixel > 100:

def estimate_label(rgb_image): # Standardized RGB image
  if avg_red(rgb_image) > avg_green(rgb_image):
    return [1,0,0] # Classify as red
  return [0,0,1] # Classify as green
