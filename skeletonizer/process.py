# coding: utf-8

from __future__ import print_function
from __future__ import division
# import matplotlib.pyplot as plt
import numpy as np
import cv2
# import opencvutils as cvu


def auto_canny(image, sigma = 0.35):
    # compute the mediam of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) *v))
    edged = cv2.Canny(image, lower, upper)

    # return edged image
    return edged

def skeletonize(image,size=(3,3)):
    # return a thin skeleton of thresholded shape
    image_size = np.size(image)
    skel = np.zeros(image.shape,np.uint8)
 
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,size)
    
    while(True):
        eroded = cv2.erode(image,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(image,temp)
        skel = cv2.bitwise_or(skel,temp)
        image = eroded.copy()
        
        zeros = image_size - cv2.countNonZero(image)
        if zeros==image_size:
            return skel
    
def load_image(url):
    """ reads url and returns cv2 image """
    return cvu.url_to_image(url)

def convert_image(image,invert=False):
    """ convert BGR image to Binary """
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    threshold_method=cv2.THRESH_BINARY+cv2.THRESH_OTSU
    if invert:
        threshold_method=cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU
    cutoff, mask=cv2.threshold(gray, 0,255, threshold_method)
    return mask

# plt.subplot(1,3,1)
# plt.imshow(logo,cmap='gray')
