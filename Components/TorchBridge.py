from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

def cv_2_tensor(img, dtype=np.float32, flagCuda=False):
    if ( 3 == len(img.shape) ):
        t = torch.from_numpy(img.astype(dtype)).permute(2,0,1).unsqueeze(0)

        if ( flagCuda ):
            t = t.cuda()
        
        return t
    elif ( 2 == len(img.shape) ):
        t = torch.from_numpy(img.astype(dtype)).unsqueeze(0).unsqueeze(0)

        if ( flagCuda ):
            t = t.cuda()

        return t
    else:
        raise Exception("img.shape must have length 3 or 2. len(img.shape) = {}. ".format(len(img.shape)))