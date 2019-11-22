from __future__ import print_function

import torch
import Corr2D_ext

def test_small_size():
    # Random tensor.
    x0 = torch.autograd.Variable( torch.rand((2,2,4,4)).cuda(), requires_grad=True )

    # Apply funcion.
    y = Corr2D_ext.test_from_BCHW_2_BHWC_padded(x0, 2)

    print(y)

    # Check.
    z = y.permute((0,3,1,2))
    x1 = z[:,:,2:6, 2:6]

    res = x1 - x0
    
    print(x1)
    print(res)

def test_reasonable_size(B=8, C=128, H=256, W=256, P=2):

    # Random tensor.
    x0 = torch.autograd.Variable( torch.rand((B, C, H, W)).cuda(), requires_grad=True )

    # Apply funcion.
    y = Corr2D_ext.test_from_BCHW_2_BHWC_padded(x0, P)

    # Check.
    z = y.permute((0,3,1,2))

    hp = z.size()[2]
    wp = z.size()[3]

    x1 = z[:,:,P:hp-P, P:wp-P]

    res = x1 - x0

    print( "test_reasonable_size: res.sum() = {}. ".format(res.sum()) )

if __name__ == "__main__":
    print("Test from_BCHW_2_BHWC_padded_cuda(). ")

    test_small_size()

    test_reasonable_size()
