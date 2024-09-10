# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image

def ncc(to_shift, anchor):
    mean_to_shift, mean_anchor = np.mean(to_shift), np.mean(anchor)

    part1 = np.sum((to_shift - mean_to_shift) * (anchor - mean_anchor))
    part2 = np.sqrt(np.sum((to_shift - mean_to_shift) ** 2) * np.sum((anchor - mean_anchor) ** 2))

    return part1 / part2

def align(to_shift, anchor):
    min_diff_shift, metric = [0, 0], -2 ** 32
    
    for r in range(-30, 31):
        for c in range(-30, 31):
            shifted_img = np.roll(to_shift, (r, c), axis=(0, 1))

            x = structural_similarity(shifted_img, anchor, data_range=1.0)

            if x > metric:
                metric = x
                min_diff_shift[0], min_diff_shift[1] = r, c

    shifted = np.roll(to_shift, (min_diff_shift[0], min_diff_shift[1]), axis=(0, 1))
    
    dim1, dim2 = shifted.shape[0], shifted.shape[1]
    dim1_range, dim2_range = (int(0.15 * dim1), int(0.85 * dim1)), (int(0.15 * dim2), int(0.85 * dim2))

    dim1_lower = dim1_range[0]
    dim2_lower = dim2_range[0]

    shifted = shifted[dim1_lower:dim1-dim1_lower, dim2_lower:dim2-dim2_lower]
    return [min_diff_shift, shifted]

def build_pyramid(m1): # returns a list of fine -> coarse images (matrices)
    pyramid = [m1]
    
    for _ in range(4):
        blurred_m1 = gaussian_filter(pyramid[-1], sigma=1)
        pyramid.append(blurred_m1[::2, ::2])
    
    return pyramid

def gaussian_pyramid_alg(to_shift, anchor, level):
    # print(to_shift.shape)
    
    to_shift_pyramid, anchor_pyramid = build_pyramid(to_shift), build_pyramid(anchor) # pyramids for g/r and b, respectively

    min_diff_shift = [0, 0]
    curr_to_shift, curr_anchor = None, None

    for i in range(level, -1, -1): # last one is most coarse
        if i == level:
            curr_to_shift, curr_anchor = to_shift_pyramid[i], anchor_pyramid[i]
            min_diff_shift = align(curr_to_shift, curr_anchor)[0] # this aligns the coarsest images and returns the min shift
        else:
            min_diff_shift = np.multiply(min_diff_shift, 2)

    return min_diff_shift

# ['emir.tif', 'monastery.jpg', 'church.tif', 'three_generations.tif', 'melons.tif', 'onion_church.tif', 
# 'train.tif', 'tobolsk.jpg', 'icon.tif', 'cathedral.jpg', 'self_portrait.tif', 'harvesters.tif', 'sculpture.tif', 'lady.tif']

# for img in listdir('../media'):
for img in ["svik724.github.io/1/extra1.tif", "svik724.github.io/1/extra2.tif"]:
    #imname = '../media/' + img
    imname = img
    im = skio.imread(imname)
    im = sk.img_as_float(im)
        
    height = np.floor(im.shape[0] / 3.0).astype(np.int64)

    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    
    level = 0 if imname.split(".")[-1] == 'jpg' else 3

    min_shift_g, min_shift_r = gaussian_pyramid_alg(g, b, level), gaussian_pyramid_alg(r, b, level)
    final_g, final_r = np.roll(g, (min_shift_g[0], min_shift_g[1]), axis=(0, 1)), np.roll(r, (min_shift_r[0], min_shift_r[1]), axis=(0, 1))

    im_out_pyramid = np.dstack([final_r, final_g, b])

    file_name = imname.split(".")[-2].split("/")[-1]
    fname = f'../output/{file_name}.tif'

    skio.imsave(fname, im_out_pyramid)

    print(img, " green: ", min_shift_g, " red: ", min_shift_r)