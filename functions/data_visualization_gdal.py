import numpy as np

import matplotlib.colors as mcolors
import matplotlib
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from osgeo import gdal
import imageio.v2 as imageio
import matplotlib.pyplot as plt
# from functions.tf_data import *
# from functions.model import *



def compare_from_pth(msk_pth, img_pth, msk_opacity=0.3, img_brightness_factor=1.5, season = False, period = 'summer'):
    if season:
        if period == 'summer':
            bands = [0,2,4]
        if period == 'winter':
            bands = [1,3,5]
    else:        
        bands = [0,1,2]
    
    ds_mask = gdal.Open(msk_pth)
    arr_mask = ds_mask.ReadAsArray()

    # Load the image
    img = imageio.imread(img_pth)

    # Apply brightness adjustment to the image
    brightened_img = img * img_brightness_factor
    brightened_img[brightened_img > 255] = 255  # Ensure values are within [0, 255]

    # Create a figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 16))

    # Plot the mask image
    axs[0].imshow(arr_mask, cmap='viridis')
    axs[0].set_title('Mask Image')
    axs[0].axis('off')

    # Plot the original image
    axs[1].imshow(brightened_img[:, :, bands])
    axs[1].set_title('Original Image')
    axs[1].axis('off')

    # Overlay the mask on the brightened image
    axs[2].imshow(brightened_img[:, :, bands])
    axs[2].imshow(arr_mask, alpha=msk_opacity)
    axs[2].set_title('Overlay')
    axs[2].axis('off')

    # Show the figure
    plt.show()

    print(f'Mask classes: {np.unique(arr_mask)}')
    print(f'Mask shape: {arr_mask.shape}')
    print(f'Image shape: {img.shape}')

    
