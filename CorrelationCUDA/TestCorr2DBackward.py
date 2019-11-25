from __future__ import print_function

import torch
import Corr2D_ext

def single_point_correlation(input0, input1, padding, kernelSize, maxDisplacement, strideK, strideD, \
        idxOutB, idxOutC, idxOutY, idxOutX):
    H = input0.size()[2]
    W = input0.size()[3]

    gridRadius = maxDisplacement // strideD

    # Kernel radius.
    kernelRadius = kernelSize // 2

    # The upper-left corner of the kernel on input0.
    y00 = idxOutY * strideK - kernelRadius + padding                   - padding
    x00 = idxOutX * strideK - kernelRadius + padding + maxDisplacement - padding

    # The bottom-right corner.
    y01 = y00 + kernelSize - 1
    x01 = x00 + kernelSize - 1

    if ( y00 < 0 ):
        y00 = 0

    if ( x00 < 0 ):
        x00 = 0

    if ( y01 >= H ):
        y01 = H - 1

    if ( x01 >= W ):
        x01 = W - 1

    # The upper-left corner of the kernel in input1.
    y10 = y00 # Same with input0.
    x10 = x00 - gridRadius * strideD + idxOutC * strideD
    
    # The bottom-right corner.
    y11 = y01 # Same with input0.
    x11 = x10 + kernelSize - 1

    if ( x10 < 0 ):
        x10 = 0

    if ( x11 >= W ):
        x11 = W - 1

    # Manually correlation.
    res = input0[idxOutB, :, y00:y01+1, x00:x01+1] * \
          input1[idxOutB, :, y10:y11+1, x10:x11+1]

    res = res.sum() / ( kernelSize**2 * input0.size()[1] )

    return res

def test_backward(B=8, C=128, H=256, W=256, \
        padding=1, kernelSize=3, maxDisplacement=64, strideK=1, strideD=1):

    print("test_backward()")

    # Random tensor.
    t0 = torch.autograd.Variable( torch.rand((B, C, H, W)).float().cuda(), requires_grad=True )
    t1 = t0.clone().detach()

    # Apply funcion.
    out = Corr2D_ext.forward( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    # Gradient.
    grad = torch.autograd.Variable( torch.ones( (B, maxDisplacement+1, H, W) ).float().cuda() )

    # Compute the autograd.
    Corr2D_ext.backward( grad, t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

def test_backward_small( \
        padding=1, kernelSize=3, maxDisplacement=1, strideK=1, strideD=1):

    print("test_backward_small()")

    B=2
    C=2
    H=8
    W=8

    # Random tensor.
    t0 = torch.autograd.Variable( torch.linspace(0, B*C*H*W-1, B*C*H*W ).view((B, C, H, W)).float().cuda(), requires_grad=True )
    t1 = t0.clone().detach()

    # Apply funcion.
    out = Corr2D_ext.forward( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

if __name__ == "__main__":
    print("Test shared memory. ")

    test_backward()
    # print("===")
    # test_backward_small()

