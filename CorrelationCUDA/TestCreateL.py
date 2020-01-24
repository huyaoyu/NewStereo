from __future__ import print_function

import torch
import Corr2D_ext

def test_small_size():
    print("test_small_size(). ")

    # Random tensor.
    x0 = torch.ones((2,2,3,5)).cuda()
    r0 = Corr2D_ext.test_from_BCHW_2_BHWC_padded(x0, 4)

    # Apply funcion.
    y = Corr2D_ext.test_create_L(r0, 3)

    print(r0[:,:,:,0])
    print("")
    print(y)

    # Check.

def test_reasonable_size_ones(B=8, C=128, H=256, W=256):
    print("test_reasonable_size_ones(). ")

    # Random tensor.
    x0 = torch.ones((B,C,H,W)).cuda()
    r0 = Corr2D_ext.test_from_BCHW_2_BHWC_padded(x0, 1)

    # Apply funcion.
    y = Corr2D_ext.test_create_L(r0, 3)

if __name__ == "__main__":
    test_small_size()

    test_reasonable_size_ones()
