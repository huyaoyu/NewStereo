from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Relative module.
from TorchBridge import cv_2_tensor
from SSIM import SSIM

# CommonPython package.
from CommonPython.ImageWrite.ImageWrite import \
    write_float_image_normalized, write_float_image_normalized_clip, write_float_image_plt, write_float_image_plt_clip

# This is copied from
# https://discuss.pytorch.org/t/measuring-peak-memory-usage-tracemalloc-for-pytorch/34067/10
def b2mb(x): return int(x/2**20)

class TorchTracemalloc():
    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        self.end  = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used   = b2mb(self.end-self.begin)
        self.peaked = b2mb(self.peak-self.begin)
        print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

if __name__ == "__main__":
    # Test on real images.
    imgFn_0 = "/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_L_VC.png"
    imgFn_1 = "/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_R_VC.png"

    # Open the two images.
    img_0 = cv2.imread(imgFn_0, cv2.IMREAD_UNCHANGED)
    img_1 = cv2.imread(imgFn_1, cv2.IMREAD_UNCHANGED)

    print("Read \n%s and\n%s" % ( imgFn_0, imgFn_1 ))

    # Convert the image into torch Tensor.
    img_0 = cv_2_tensor(img_0, dtype=np.float32, flagCuda=True)
    img_1 = cv_2_tensor(img_1, dtype=np.float32, flagCuda=True)

    with TorchTracemalloc() as tt:
        
        # SSIM.
        ssim3 = SSIM(channel=3)
        res = ssim3(img_0, img_1)

        print("res.size() = {}.".format(res.size()))
        res = res.squeeze(0).squeeze(0).cpu().numpy()

        # Statistics.
        print("res: mean = {}, max = {}, min = {}.".format( res.mean(), res.max(), res.min() ))

        res2 = ssim3(img_0, img_1)

    print("GPU memory used = %fMB, peaked = %fMB. " % ( tt.used, tt.peaked ))
