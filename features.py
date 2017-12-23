import numpy as np
import cv2

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

def high_saturation_pixels(rgb_image):
    # Returns average red and green content from high saturation pixels
  high_saturation_pixels = []
  saturation_threshold = 100
  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
  for i in range(32):
    for j in range(32):
      if hsv[i][j][1] > saturation_threshold:
        high_saturation_pixels.append(rgb_image[i][j])

  if not high_saturation_pixels:
    return 0, 0

  sum_red = 0
  sum_green = 0
  for pixel in high_saturation_pixels:
    sum_red += pixel[0]
    sum_green += pixel[1]
  # print(sum_red)
  # print(sum_green)

  # print(high_saturation_pixels[0])
  # print(np.sum(high_saturation_pixels[0])) # Sum of red pixels
  avg_red = sum_red / len(high_saturation_pixels)
  avg_green = sum_green / len(high_saturation_pixels)
  return avg_red, avg_green

def estimate_label(rgb_image): # Standardized RGB image
  red, green = high_saturation_pixels(rgb_image)
  if red > green:
    return [1,0,0] # Classify as red
  return [0,0,1] # Classify as green
