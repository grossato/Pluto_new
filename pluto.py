"""
Title: 'pluto.py'
Author: Curcuraci L.
Date: 25/05/21

Scope: Sketch a possible semi-automatic segmentation algorithm.
"""


#################
#####   LIBRARIES
#################


import numpy as np
import matplotlib.pyplot as plt
from roipoly import RoiPoly
from skimage.measure import regionprops
from scipy.ndimage.morphology import binary_dilation,binary_erosion,binary_fill_holes

from bmip_tools.stack import Stack
from bmip_tools.visualization.graphic_tools.basic_graphic_tools import Basic2D as b2d


#################
#####   FUNCTIONS
#################


def draw_mask_on_image(image,title='',cmap='viridis',zoomed_boundary = None):
    """
    Draw a mask on an 2D image and get the binary mask from it.

    :param image: image on which the mask is drawn.
    :param title: (optional) title of the image.
    :param cmap: (optional) colormap used for the image.
    :return: binary mask of the drawn mask.
    """
    mask = np.zeros(image.shape,dtype=np.uint8)
    image_vis = image
    if zoomed_boundary is not None:

        image_vis = image[zoomed_boundary[0]:zoomed_boundary[1],zoomed_boundary[2]:zoomed_boundary[3]]

    plt.figure()
    plt.title(title)
    plt.imshow(image_vis,cmap=cmap)
    roi = RoiPoly(color='r')
    plt.show()
    pre_mask = roi.get_mask(image_vis)
    if zoomed_boundary is not None:

        mask[zoomed_boundary[0]:zoomed_boundary[1],zoomed_boundary[2]:zoomed_boundary[3]] = pre_mask
        return mask

    return pre_mask


############
#####   MAIN
############


path_to_dataset = r'pluto_ct/cropped_3_3'

stack = Stack(path_to_dataset,from_folder=True,load_metadata=False)
data = stack.data


##### Pluto algorithm










# User input
N_iterations_max = 40                                      # max number of iterations in the grow of the candidate mask.
alpha = 0.8                                                # moving average for mean and standard deviation update.
N_slice_max = 20

# Draw the initial mask.
b2d.show_image(data[0])                                                     # uncomment to see where to zoom to draw the
                                                                            # mask better. The region to be considered
                                                                            # expressed in numpy as y1:y2,x1:x2 have to
                                                                            # be expressed as [y1,y2,x1,x2] in the
                                                                            # 'zoomed_boundary' variable of the
                                                                            # 'draw_mask_on_image' function.
mask_slice0 = draw_mask_on_image(data[0],zoomed_boundary=[85,105,105,130])
b2d.show_threshold_on_image(data[0],mask_slice0)

# Step 1: characterize the starting mask (mean,std,centroid coordinates)
reg = regionprops(mask_slice0)
yc,xc = list(map(int,reg[0].centroid))
mean = np.mean(data[0][np.where(mask_slice0 == 1)])
std = np.std(data[0][np.where(mask_slice0 == 1)])

# Step 2-N_slices: create a mask from a seed point (or more) in the next slice.
masks = [mask_slice0]
for N in range(1,N_slice_max):

    print('slice {}/{} | centroid: ({},{})'.format(N,N_slice_max,yc,xc))
    slice = data[N]
    mask_slice = np.zeros(shape=mask_slice0.shape,dtype=np.uint8)
    mask_slice[yc,xc] = 1                                        # single seed point: the centroid (super easy choice)
    reg_tmp = regionprops(mask_slice)
    Candidate_region_area = reg_tmp[0].area                      # area is used to stop the cycle.
    for i in range(N_iterations_max):

        tmp_pre_mask = binary_dilation(mask_slice,iterations=1)
        points_to_exclude = np.logical_or(slice[tmp_pre_mask] < mean-2*std,slice[tmp_pre_mask] > mean+2*std)
        region_ys,region_xs = np.where(tmp_pre_mask == 1)
        tmp_pre_mask[region_ys[points_to_exclude],region_xs[points_to_exclude]] = 0
        reg_tmp = regionprops(tmp_pre_mask.astype(int))

        # if the area stop to grow the growing-cycle is interrupted earlier
        if reg_tmp[0].area > Candidate_region_area:

            mask_slice = tmp_pre_mask
            Candidate_region_area = reg_tmp[0].area
            continue

        else:

            break

    reg_new_mask = regionprops(mask_slice.astype(int))
    yc,xc = list(map(int,reg_new_mask[0].centroid))

    # exponential moving average to update the defining parameters defining the canalicula region slowly (increase
    # robustness to possible estimation errors in a single slice)
    mean = alpha*mean +(1-alpha)*np.mean(data[N][np.where(mask_slice == 1)])
    # std = std
    std = alpha*std + (1-alpha)*np.std(data[N][np.where(mask_slice == 1)])

    # fill the holes if present....necessary?
    mask_slice = binary_fill_holes(mask_slice)

    masks.append(mask_slice)

# C
# heck the segmentation result (the results are the one with N>0)
N = 18
b2d.show_threshold_on_image(data[N],masks[N])

"""
Possible problems and observations

- selection of the seed point too rigid: maybe by randomly picking a fraction of the point in the previous mask, one 
  can find better seed point, which may solve two possible problems:
  
                    > follow better the canalicula if it curves;
                    
                    > avoid to start to grow in regions of the canalicula where some occlusion may be present, or
                      in regions which are not part of it, simply because the canalicula is changing shape.

- different slices have different average brightness. Maybe matching the histogram and standardizing all the images may 
  help;

- moving average is used in order to avoid that mean and standard deviations changes to fast, but in principle it should
  not be necessary. Why the mean and standard deviations changes too much? Why do not use weights to penalize points in 
  the mask which are to far away from the mean value computed in the previous slice (e.g. wi = A/|pixel_val-mean| where
  A such that sum_i w_i =1)?
  
- Tested with somehow acceptable result only for 20 slice...probably with more slices more wild behaviour may happens

- Problem with the code: if the mask is not found an error is returned and everything stop (this may happens if the 
  parameters does not allow the mask to grow, i.e. it remains as a single point).

"""