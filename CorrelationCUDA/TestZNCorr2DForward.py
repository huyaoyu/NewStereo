from __future__ import print_function

import numpy as np

import torch
import Corr2D_ext


def single_point_correlation(input0, input1, padding, kernelSize, maxDisplacement, strideK, strideD, \
        idxOutB, idxOutC, idxOutY, idxOutX):

    gridRadius = maxDisplacement // strideD

    # Kernel radius.
    kernelRadius = kernelSize // 2

    # Padding.
    paddingModule = torch.nn.ConstantPad2d(padding, 0)
    input0 = paddingModule( input0 )
    input1 = paddingModule( input1 )

    H = input0.size()[2]
    W = input0.size()[3]

    # The upper-left corner of the kernel on input0.
    y00 = idxOutY * strideK - kernelRadius + gridRadius*strideD
    x00 = idxOutX * strideK - kernelRadius + gridRadius*strideD

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

    if ( x10 < 0 ):
        return 0
    
    # The bottom-right corner.
    y11 = y01 # Same with input0.
    x11 = x10 + kernelSize - 1

    if ( x10 < 0 ):
        x10 = 0

    if ( x11 >= W ):
        x11 = W - 1

    # Manually correlation.
    kernel0 = input0[idxOutB, :, y00:y01+1, x00:x01+1]
    kernel1 = input1[idxOutB, :, y10:y11+1, x10:x11+1]
    nk0 = kernel0.norm(p=2)
    nk1 = kernel1.norm(p=2)

    if ( nk0 == 0 or nk1 == 0 ):
        return 0

    L0 = 1.0 / nk0
    L1 = 1.0 / nk1

    # import ipdb; ipdb.set_trace()

    res = kernel0 * kernel1

    res = res.sum() * L0 * L1

    return res

def test_forward(B=8, C=128, H=256, W=256, \
        padding=64, kernelSize=3, maxDisplacement=64, strideK=1, strideD=1):

    print("test_forward()")

    # Random tensor.
    t0 = torch.rand((B, C, H, W)).double().cuda()
    t1 = t0.clone().detach()

    # Apply funcion.
    out, L0, L1 = Corr2D_ext.forward( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    print("out[0, :, 0, 0] = \n{}".format(out[0, :, 0, 0]))

    # Manually perform a correlation for out[0, -1, 0, 0].
    res = single_point_correlation( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD, \
        0, out.size()[1]-1, 0, 0 )

    print("res = {}. ".format(res))
    print("res - out[0, -1, 0, 0] = {}. ".format( res - out[0, -1, 0, 0] ))

     # Manually perform a correlation for out[0, -2, 0, 0].
    res = single_point_correlation( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD, \
        0, out.size()[1]-2, 0, 0 )

    print("res = {}. ".format(res))
    print("res - out[0, -2, 0, 0] = {}. ".format( res - out[0, -2, 0, 0] ))

    # import ipdb; ipdb.set_trace()
    # Manually perform a correlation for out[0, 0, 0, 1].
    res = single_point_correlation( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD, \
        0, 0, 0, 1 )

    print("out[0, :, 0, 1] = \n{}".format( out[0, :, 0, 1] ))
    print("res = {}. ".format(res))
    print("res - out[0, 0, 0, 1] = {}. ".format( res - out[0, 0, 0, 1] ))

    # Manually perform a correlation for out[0, 0, 0, 64].
    res = single_point_correlation( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD, \
        0, 0, 0, 64 )

    print("out[0, :, 0, 64] = \n{}".format( out[0, :, 0, 64] ))
    print("res = {}. ".format(res))
    print("res - out[0, 0, 0, 64] = {}. ".format( res - out[0, 0, 0, 64] ))

def test_forward_range( \
        padding=64, kernelSize=3, maxDisplacement=64, strideK=1, strideD=1):

    print("test_forward_range()")

    a = np.array( range(256, 0, -1) )
    b = np.stack( (a, a, a), axis=0 )

    # Random tensor.
    t0 = torch.from_numpy( b ).double().unsqueeze(0).unsqueeze(0)
    t1 = t0.clone().detach()

    t0 = t0.cuda()
    t1 = t1.cuda()

    # Apply funcion.
    out, L0, L1 = Corr2D_ext.forward( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    print("out[0, :, 0, 0] = \n{}".format(out[0, :, 0, 0]))

    # import ipdb; ipdb.set_trace()

    # Manually perform a correlation for out[0, -1, 0, 0].
    res = single_point_correlation( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD, \
        0, out.size()[1]-1, 0, 0 )

    print("res = {}. ".format(res))
    print("res - out[0, -1, 0, 0] = {}. ".format( res - out[0, -1, 0, 0] ))

    # Manually perform a correlation for out[0, 0, 0, 1].
    res = single_point_correlation( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD, \
        0, 0, 0, 1 )

    print("out[0, :, 0, 1] = \n{}".format( out[0, :, 0, 1] ))
    print("res = {}. ".format(res))
    print("res - out[0, 0, 0, 1] = {}. ".format( res - out[0, 0, 0, 1] ))

    # Manually perform a correlation for out[0, 0, 0, 64].
    res = single_point_correlation( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD, \
        0, 0, 0, 64 )

    print("out[0, :, 0, 64] = \n{}".format( out[0, :, 0, 64] ))
    print("res = {}. ".format(res))
    print("res - out[0, 0, 0, 64] = {}. ".format( res - out[0, 0, 0, 64] ))

def test_forward_small( \
        padding=1, kernelSize=3, maxDisplacement=1, strideK=1, strideD=1):

    print("test_forward_small()")

    B=2
    C=2
    H=8
    W=8

    # Random tensor.
    t0 = torch.autograd.Variable( torch.linspace(0, B*C*H*W-1, B*C*H*W ).view((B, C, H, W)).float().cuda(), requires_grad=True )
    t1 = t0.clone().detach()

    # Apply funcion.
    out, L0, L1 = Corr2D_ext.forward( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    print("out[0, :, 0, 0] = \n{}".format( out[0, :, 0, 0] ))

    # Manually perform a correlation for out[0, -1, 0, 0].
    res = single_point_correlation( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD, \
        0, out.size()[1]-1, 0, 0 )

    print("res = {}. ".format(res))
    print("res - out[0, -1, 0, 0] = {}. ".format( res - out[0, -1, 0, 0] ))

    # Manually perform a correlation for out[0, 0, 0, 1].
    res = single_point_correlation( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD, \
        0, 0, 0, 1 )

    print("out[0, :, 0, 1] = \n{}".format( out[0, :, 0, 1] ))
    print("res = {}. ".format(res))
    print("res - out[0, 0, 0, 1] = {}. ".format( res - out[0, 0, 0, 1] ))

if __name__ == "__main__":
    test_forward()
    print("===")
    test_forward_small()
    print("===")
    test_forward_range()

