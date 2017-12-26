import numpy as np
import cv2


def create_feature(rgb_image):

  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV) # Convert to HSV color space

  sum_brightness = np.sum(hsv[:,:,2]) # Sum the brightness values
  area = 32*32
  avg_brightness = sum_brightness / area # Find the average

  return avg_brightness

def high_saturation_pixels(rgb_image, threshold):
  # Returns average red and green content from high saturation pixels
  high_sat_pixels = []
  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
  for i in range(32):
    for j in range(32):
      if hsv[i][j][1] > threshold:
        high_sat_pixels.append(rgb_image[i][j])

  if not high_sat_pixels:
    return highest_sat_pixel(rgb_image)

  sum_red = 0
  sum_green = 0
  for pixel in high_sat_pixels:
    sum_red += pixel[0]
    sum_green += pixel[1]

  # TODO: Use sum() instead of manually adding them up
  avg_red = sum_red / len(high_sat_pixels)
  avg_green = sum_green / len(high_sat_pixels) * 0.8 # 0.8 to favor red's chances
  return avg_red, avg_green

def highest_sat_pixel(rgb_image):
  '''Finds the higest saturation pixel, and checks if it has a higher green
  content, or a higher red content'''

  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
  s = hsv[:,:,1]

  x, y = (np.unravel_index(np.argmax(s), s.shape))
  if rgb_image[x, y, 0] > rgb_image[x,y, 1] * 0.9: # 0.9 to favor red's chances
    return 1, 0 # Red has a higher content
  return 0, 1

def estimate_label(rgb_image): # Standardized RGB image
  saturation_threshold = 80
  red, green = high_saturation_pixels(rgb_image, saturation_threshold)
  if red > green:
    return [1,0,0] # Classify as red
  return [0,0,1] # Classify as green
