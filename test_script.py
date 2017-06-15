#!/usr/bin/env

import sys, warnings
import numpy as np
from timeit import default_timer as timer

import skeletonizer.process as skp
import skimage.filters, skimage.morphology, skimage.feature, skimage.io
import skimage.transform, skimage.util

import sklearn.metrics.pairwise
import sklearn.feature_extraction.image

timer_start = timer()
def print_timer(timer_start,msg=""):
    end = timer()
    print(">> TIMER %0.4fs %s"%(end - timer_start,msg))
    return end

def find_sigma(gray, limit):
    """ Return sigma value, where less than limit pixels have canny edge, max=5 """
    image_area=np.prod(gray.shape)
    limit=int(limit*image_area)
    for sigma in np.arange(0,5,0.05):
        canny=skp.auto_canny(gray,sigma=sigma)
        if np.sum(canny) < limit:
            break
        # find level where there are no edges in a control area
    return sigma

def find_level(gray, limit):
    """ Return threshold value, where less than limit pixels have bright value """
    image_area=np.prod(gray.shape)
    limit=int(limit*image_area)
    for level in np.arange(0,255,1):
        mask=gray>level
        if np.sum(mask) < limit:
            break
        # find level where there are no mask pixels in a control area
    return level

def find_levels(gray, limit=1):
    """ Return min and max threshold values. limit is percentiles """
    return ( np.percentile(gray,limit), 
             np.percentile(gray,100-limit) 
           )

def get_lbp_histogram(gray):
    radius=3
    no_points=8*radius
    lbp=skimage.feature.local_binary_pattern(gray, no_points, radius, method='uniform')
    freq=np.histogram(lbp.ravel())[0]
    hist = freq/sum(freq)
    return hist
    
def get_patch_lbp(gray,no_patches=64):
    histograms=[]
    for patch in sklearn.feature_extraction.image.extract_patches_2d(gray, (32,32),max_patches=no_patches):
        histograms.append(get_lbp_histogram(patch))
    return np.array(histograms)

def compare_lbps(gray,cntrl):
    distances=[]
    for i,patch in enumerate(sklearn.feature_extraction.image.extract_patches_2d(gray, (32,32))):
        print(i)
        if i>10:
            break
        distances.append(
            sklearn.metrics.pairwise.chi2_kernel(
                cntrl,
                get_lbp_histogram(patch).reshape(1,-1)
            )
        )
    return np.array(distances)

def resize_image(image):
    if (max(image.shape)>512):
        # image too large:
        aspect_ratio= float(image.shape[0])/float(image.shape[1])
        new_width=512
        new_height = int(aspect_ratio*new_width)
        image = skimage.transform.resize(image, (new_height,new_width),mode='edge')
    return image

#url='http://imaging.ninja/img/anima.png'
#url='http://hbu.h-cdn.co/assets/15/41/768x514/gallery-1444338501-eiffel-tower-at-night.jpg'
url="boat.jpg"
invert=False
print("Invert",invert)
if len(sys.argv)>1:
    url=sys.argv[1]

gray=skimage.filters.median(resize_image(skp.load_image(url)),
            selem=skimage.morphology.square(3))
if invert:
    gray=skimage.util.invert(gray)
timer_start=print_timer(timer_start,"Image loaded")
corner_size=(64,128)
#~ [int(float(x)/4) for x in gray.shape]
corner_area=np.prod(corner_size)
print("Corner",corner_size, corner_area)

corner1=gray[-corner_size[0]:gray.shape[0], 0:corner_size[1]]
corner2=gray[-corner_size[0]:gray.shape[0], -corner_size[1]:gray.shape[1]]

sigma1=find_sigma(corner1, 0.0003)
sigma2=find_sigma(corner2, 0.0003)
sigma=np.max((sigma1,sigma2))
print("Sigma",sigma1,sigma2)

level1=find_level(corner1,0.01)
level2=find_level(corner2,0.01)

levels=find_levels((corner1,corner2))
print("Level",levels)

corner_lbps=get_patch_lbp(corner1)
corner_lbps=np.append(corner_lbps,get_patch_lbp(corner2),axis=0)
print(corner_lbps.shape)
print(np.mean(corner_lbps,0))

timer_start=print_timer(timer_start,"Normalization done")

mask=np.logical_or(gray<levels[0],gray>levels[1])
canny=[1,0,0]*skimage.color.gray2rgb(skp.auto_canny(gray,sigma=sigma))
skeleton=[0,1,0]*skimage.color.gray2rgb(skp.skeletonize(mask))
merger=canny+skeleton

random_patches_idx=np.random.randint(0,np.prod(gray.shape),2048)
random_patches=[]
for idx in random_patches_idx:
    x=int(idx/gray.shape[0])
    y=idx%gray.shape[0]
    try:
        random_patches.append((get_lbp_histogram(gray[ (y-16):(y+16), (x-16):(x+16) ]),y,x) )
    except ValueError:
        pass
random_patches=np.array(random_patches)
distances=sklearn.metrics.pairwise.chi2_kernel(
                corner_lbps,
                [x[0] for x in random_patches])
min_distances=distances.min(0)
distance_heatmap=np.zeros(gray.shape,dtype=np.uint8)
for patch in zip(random_patches,min_distances):
    distance_heatmap[ patch[0][1], patch[0][2] ]=255*patch[1]
distance_heatmap=skimage.filters.rank.maximum(distance_heatmap,
                        selem=skimage.morphology.square(int(32/2)))
heatmap_color=skimage.color.gray2rgb(distance_heatmap)
heatmap_color[::,::,1]=255
heatmap_color[::,::,2]=128
heatmap_color=skimage.color.hsv2rgb(heatmap_color)
heatmap_color=skimage.color.gray2rgb(distance_heatmap>0)*heatmap_color
heatmap_visu=skimage.color.gray2rgb(gray)
heatmap_visu[heatmap_color>0]=heatmap_color[heatmap_color>0]
print(random_patches.shape)
#lbps_comparison=compare_lbps(gray,corner_lbps)
#print(lbps_comparison)

import matplotlib.pyplot as plt
plt.imshow(distance_heatmap,cmap='gray')
plt.draw()
plt.pause(0.1)
for r in random_patches:
    pass
    #~ print(r)
    #~ plt.imshow(r)
    #~ plt.draw()
    #~ plt.pause(0.1)

timer_start=print_timer(timer_start,"Detection done")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if invert:
        gray=skimage.util.invert(gray)
        gray[-corner_size[0]:gray.shape[0], 0:corner_size[1]]-=corner1
        gray[-corner_size[0]:gray.shape[0], -corner_size[1]:gray.shape[1]]-=corner2
    else:
        gray[-corner_size[0]:gray.shape[0], 0:corner_size[1]]+=corner1
        gray[-corner_size[0]:gray.shape[0], -corner_size[1]:gray.shape[1]]+=corner2
    skimage.io.imsave('01-gray.png',gray)
    skimage.io.imsave('02-mask.png',255*mask)
    skimage.io.imsave('03-canny.png',255*canny)
    skimage.io.imsave('04-skeleton.png',255*skeleton)
    skimage.io.imsave('05-merger.png',255*merger)


# plt.subplot(2,2,1)
# plt.imshow(gray,cmap='gray')
# plt.subplot(2,2,2)
# plt.imshow(mask,cmap='gray')
# plt.subplot(2,2,3)
# plt.imshow(canny)
# plt.subplot(2,2,4)
# plt.imshow(skeleton)
# plt.savefig("test_figure.png",dpi=150)


timer_start=print_timer(timer_start,"Images saved")
