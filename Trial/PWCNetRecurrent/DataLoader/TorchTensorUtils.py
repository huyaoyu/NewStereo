from __future__ import print_function

import cv2
import numpy as np
import torch
import os

def save_tensor_image(fn, x):
    """
    Save a torch tensor as an image.
    x is the tensor. x could be in shape of (C, H, W) or (H, W).
    The values in x must be already regulated in the range of 0-255, although internal clipping
    is applied.

    Only png file is supported for output.
    """

    if ( 3 == len( x.size() ) ):
        x = x.permute((1, 2, 0))
    
    # Get the CPU NumPy version.
    x = torch.clamp(x, 0, 255)
    x = x.cpu().numpy().astype(np.uint8)

    # Save the iamge.
    cv2.imwrite(fn, x, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def save_tensor_image_single_channel_normalized(fn, x, low=0.0, high=0.0):
    """
    Values in x will be clipped if they are less than 'low' or greater than 'high'.

    Only png file is supported for output.
    """

    if ( 3 == len( x.size() ) ):
        if ( 1 != x.size()[0] ):
            raise Exception("Only single channel tensor is supported. x.size() = {}. ".format( x.size() ))

        # Get the 2D version of x.
        x = x.squeeze(0)
    
    if ( low >= high ):
        # Normalize by min and max.
        x = x - x.min()
        x = x / x.max()
    else:
        x = x - low
        x = x / ( high - low )
        x = torch.clamp( x, 0, 1 )

    # Get the CPU NumPy version.
    x = x.cpu().numpy() * 255
    x = x.astype(np.uint8)

    # Save the image.
    cv2.imwrite( fn, x, [cv2.IMWRITE_PNG_COMPRESSION, 0] )