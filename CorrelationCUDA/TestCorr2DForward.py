from __future__ import print_function

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

def test_forward(B=8, C=128, H=256, W=256, \
        padding=1, kernelSize=3, maxDisplacement=64, strideK=1, strideD=1):

    print("test_forward()")

    # Random tensor.
    x0 = torch.autograd.Variable( torch.rand((B, C, H, W)).float().cuda(), requires_grad=True )
    x1 = x0.clone().detach()

    # Apply funcion.
    y = Corr2D_ext.forward( x0, x1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    print("y[0, :, 0, 0] = {}".format(y[0, :, 0, 0]))

def test_forward_small( \
        padding=1, kernelSize=3, maxDisplacement=1, strideK=1, strideD=1):

    print("test_forward_small()")

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

if __name__ == "__main__":
    print("Test shared memory. ")

    # test_shared_memory()
    # test_shared_memory_small()

    test_forward()
    test_forward_small()

