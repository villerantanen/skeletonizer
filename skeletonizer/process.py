# coding: utf-8

from __future__ import print_function
from __future__ import division
# import matplotlib.pyplot as plt
import numpy as np
import sys
import skimage.io, skimage.filters, skimage.feature, skimage.morphology


if sys.version_info[0] == 2:
    from urllib import urlopen
else:
    from urllib.request import urlopen

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    image=skimage.io.imread(url, as_grey=True)

    # return the image
    return image

def auto_canny(image, sigma = 0.35):
    # compute the median of the single channel pixel intensities
    edged = skimage.feature.canny(image,sigma=sigma)

    # return edged image
    return edged

def skeletonize(image):
    # return a thin skeleton of thresholded shape
    if (image==0).all() or (image==255).all():
        # Image is empty, skeletonization impossible
        return np.zeros(image.shape,np.uint8)

    return skimage.morphology.skeletonize(image)

    
def load_image(url):
    """ reads url and returns cv2 image """
    return url_to_image(url)

def convert_image(image,invert=False,adjust=1.0):
    """ convert Gray to mask
        invert:  Dark pixels are object pixels
        adjust:  multily result of Otsu
    """
    cutoff=skimage.filters.threshold_otsu(image)
    cutoff=int(adjust*cutoff)
    if invert:
        mask=image<cutoff
    else:
        mask=image>cutoff
    return mask

# plt.subplot(1,3,1)
# plt.imshow(logo,cmap='gray')
