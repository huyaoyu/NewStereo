import cv2
import math
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

if ( __name__ == "__main__" ):
    import sys

    sys.path.insert(0, "/home/yaoyu/Projects/NewStereo/Trial/PRR/Model")
    from StereoUtility import WarpByDisparity as WModel
else:
    from .StereoUtility import WarpByDisparity as WModel

class RecurrentModel(object):
    def __init__(self, rModel):
        super(RecurrentModel, self).__init__()

        self.rModel = rModel
        self.warp = WModel()

    def apply(self, stack0, stack1, disp, coordinates, flagWarp=False):
        """
        stack0, stack1 must be 4D tensors, (B, C, H, W).
        disp is a 4D tensor, (B, C, H, W)
        coordinates is a 4-element list/tuple/array, [x0, y0, x1, y1].
        flagWarp set to True to let this funcion perform a warp before recurrently
        estimating the disparity.
        """
        
        HS0 = stack0.size()[2]
        WS0 = stack0.size()[3]

        HS1 = stack1.size()[2]
        WS1 = stack1.size()[3]

        HD  = disp.size()[2]
        WD  = disp.size()[3]

        x0 = coordinates[0]
        y0 = coordinates[1]
        x1 = coordinates[2]
        y1 = coordinates[3]

        # Checkt the dimensions.
        assert ( HS0 == HS1 and HS0 == HD ), "HS0 = {}, HS1 = {}, HD = {}. ".format( HS0, HS1, HD )
        assert ( WS0 == WS1 and WS0 == WD ), "WS0 = {}, WS1 = {}, WD = {}. ".format( WS0, WS1, WD )
        assert ( y0 >= 0 and x0 >= 0 ), "y0 = {}, x0 = {}. ".format( y0, x0 )
        assert ( y1 > y0 and x1 > x0 ), "y0 = {}, x0 = {}. y1 = {}, x1 = {}. ".format( y0, x0, y1, x1 )
        assert ( y1 < HS0 and x1 < WS0 ), "y1 = {}, x1 = {}, HS0 = {}, WS0 = {}. ".format( y1, x1, HS0, WS0 )

        # Initial warp.
        if ( flagWarp ):
            stack1 = self.warp(stack1, disp)

        # Get the data.
        patch0 = stack0[:, :, y0:y1+1, x0:x1+1]
        patch1 = stack1[:, :, y0:y1+1, x0:x1+1]
        patchD = disp[:, :, y0:y1+1, x0:x1+1]

        # Apply the one_shot() method for once.
        disp0, extra = self.rModel.one_shot(patch0, patch1, patchD)

        return disp0, extra

