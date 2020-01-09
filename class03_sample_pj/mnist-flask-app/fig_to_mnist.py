# The standard MNIST format are 28x28 grayscale images with 0 = black.
# Moreover, for classification to work properly the digit needs to be centred inside a 20x20 box.
# These functions take a cv2 image format (RGB or grayscale) in an arbitrary size and position
# and return an image formated as above.

from scipy import ndimage
import numpy as np
import math
import cv2

# Scales input image into a max_height x max_width grayscale image
def converter(myFig, max_height, max_width):
    height, width = myFig.shape[:2]
    if max_height < height or max_width < width:
    # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width/ float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
            # resize image
        small = cv2.resize(myFig, (20,20), fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        small = 255 - small
        rows,cols = small.shape
        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        small = np.lib.pad(small,(rowsPadding,colsPadding),'constant')
        return(small)
    else:
        print('Your image is already smaller than {} x {}'.format(max_height, max_width))
        return(myFig)

# Crop image (i.e. gets rid of part of the background)
def cropper(myFig):
    retval, thresh_gray = cv2.threshold(myFig, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

    # find where the signature is and make a cropped region
    points = np.argwhere(thresh_gray==0) # find where the black pixels are
    points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
    x, y, w, h = x-10, y-10, w+20, h+20 # make the box a little bigger
    crop = myFig[y:y+h, x:x+w] # create a cropped region of the gray image

    # get the thresholded crop
    retval, thresh_crop = cv2.threshold(crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
    return(thresh_crop)

# Find the shift to centre cropped image
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

# Shift the image by sx and sy
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

# Wrap
def mnist_treat(fig):
    max_height = 28
    max_width = 28
    fig_small = converter(cropper(fig), max_height, max_width)
    shiftx, shifty = getBestShift(fig_small)
    fig_shifted = shift(fig_small, shiftx, shifty)
    return(fig_shifted)
