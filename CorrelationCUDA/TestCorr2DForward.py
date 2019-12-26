from __future__ import print_function

import numpy as np

import torch
import Corr2D_ext

def test_shared_memory(B=8, C=128, H=256, W=256, \
        padding=1, kernelSize=3, maxDisplacement=64, strideK=1, strideD=1):

    print("test_shared_memory()")

    # Random tensor.
    x0 = torch.autograd.Variable( torch.rand((B, C, H, W)).float().cuda(), requires_grad=True )
    x1 = x0.clone().detach()

    # Apply funcion.
    y = Corr2D_ext.forward( x0, x1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    # Check.
    print("y[0, 0, 1, 1] = %f" % ( y[0, 0, 1, 1] ) )
    print("y[0, 1, 1, 1] = %f" % ( y[0, 1, 1, 1] ) )

    kernel = x0[0, :, 0:3, maxDisplacement:maxDisplacement+3]
    print("kernel.shape = {}. ".format(kernel.shape))
    print("kernel.sum() = {}. ".format(kernel.sum()))
    print("kernel.sum() - y[0,0,1,1] = {}. ".format( kernel.sum() - y[0,0,1,1] ))

def test_shared_memory_small( \
        padding=1, kernelSize=3, maxDisplacement=1, strideK=1, strideD=1):

    print("test_shared_memory_small()")

    B=2
    C=2
    H=8
    W=8

    # Random tensor.
    x0 = torch.autograd.Variable( torch.linspace(0, B*C*H*W-1, B*C*H*W ).view((B, C, H, W)).float().cuda(), requires_grad=True )
    x1 = x0.clone().detach()

    # Apply funcion.
    y = Corr2D_ext.forward( x0, x1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    # Check.
    print("y[0, 0, 1, 1] = %f" % ( y[0, 0, 1, 1] ) )
    print("y[0, 1, 1, 1] = %f" % ( y[0, 1, 1, 1] ) )

    kernel = x0[0, :, 0:3, maxDisplacement:maxDisplacement+3]
    print("kernel.shape = {}. ".format(kernel.shape))
    print("kernel.sum() = {}. ".format(kernel.sum()))
    print("kernel.sum() - y[0,0,1,1] = {}. ".format( kernel.sum() - y[0,0,1,1] ))
    print("The sum should be 756.0.")

    if ( (y[0, 0, 1, 1] - 756.0) != 0.0 ):
        raise Exception("y[0, 0, 1, 1] - 756.0 = %e. " % (y[0, 0, 1, 1] - 756.0))

    print("y[0, 0, 2, 2] = %f" % ( y[0, 0, 2, 2] ) )
    print("y[0, 1, 2, 2] = %f" % ( y[0, 1, 2, 2] ) )

    kernel = x0[0, :, 1:4, maxDisplacement+1:maxDisplacement+1+3]
    print("kernel.shape = {}. ".format(kernel.shape))
    print("kernel.sum() = {}. ".format(kernel.sum()))
    print("kernel.sum() - y[0,0,2,2] = {}. ".format( kernel.sum() - y[0,0,2,2] ))
    print("The sum should be 918.0.")

    if ( (y[0, 0, 2, 2] - 918.0) != 0.0 ):
        raise Exception("y[0, 0, 2, 2] - 918.0 = %e. " % (y[0, 0, 2, 2] - 918.0))

    print("y[0, 0, 0, 6] = %f" % ( y[0, 0, 0, 6] ) )
    print("y[0, 1, 0, 6] = %f" % ( y[0, 1, 0, 6] ) )

    kernel = x0[0, :, 0:2, maxDisplacement+5:maxDisplacement+5+2]
    print("kernel.shape = {}. ".format(kernel.shape))
    print("kernel.sum() = {}. ".format(kernel.sum()))
    print("kernel.sum() - y[0,0,0,6] = {}. ".format( kernel.sum() - y[0,0,0,6] ))
    print("The sum should be 340.0.")

    if ( (y[0, 0, 0, 6] - 340.0) != 0.0 ):
        raise Exception("y[0, 0, 2, 2] - 918.0 = %e. " % (y[0, 0, 0, 6] - 340.0))

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

def test_forward(B=8, C=128, H=256, W=256, \
        padding=1, kernelSize=3, maxDisplacement=64, strideK=1, strideD=1):

    print("test_forward()")

    # Random tensor.
    t0 = torch.autograd.Variable( torch.rand((B, C, H, W)).float().cuda(), requires_grad=True )
    t1 = t0.clone().detach()

    # Apply funcion.
    out = Corr2D_ext.forward( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    print("out[0, :, 0, 0] = \n{}".format(out[0, :, 0, 0]))

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

def test_forward_range( \
        padding=1, kernelSize=3, maxDisplacement=64, strideK=1, strideD=1):

    print("test_forward_range()")

    a = np.array( range(256, 0, -1) )
    b = np.stack( (a, a, a), axis=0 )

    # Random tensor.
    t0 = torch.from_numpy( b ).double().unsqueeze(0).unsqueeze(0)
    t1 = t0.clone().detach()

    t0 = t0.cuda()
    t1 = t1.cuda()

    # Apply funcion.
    out = Corr2D_ext.forward( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    print("out[0, :, 0, 0] = \n{}".format(out[0, :, 0, 0]))

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
    out = Corr2D_ext.forward( t0, t1, \
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
    print("Test shared memory. ")

    # test_shared_memory()
    # test_shared_memory_small()

    test_forward()
    print("===")
    test_forward_small()
    print("===")
    test_forward_range()

