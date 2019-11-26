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

    print("out.size() = {}. ".format( out.size() ))

    gridRadius = maxDisplacement // strideD

    # Gradient.
    grad = torch.ones( (B, gridRadius+1, H, W-gridRadius) ).float().cuda()

    print("grad.size() = {}. ".format(grad.size()))

    # Compute the autograd.
    Corr2D_ext.backward( grad, t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

def naive_correlation(input0, input1, \
        padding=1, kernelSize=3, maxDisplacement=1, strideK=1, strideD=1, floatType=torch.float32):
    
    kernelRadius = kernelSize // 2

    gridRadius = maxDisplacement // strideD
    gridRelativeStart = -gridRadius

    # Padding.
    s = input0.size()
    r0 = torch.zeros(( s[0], s[1], s[2]+2*padding, s[3]+2*padding ), dtype=floatType, requires_grad=True).cuda()
    r1 = torch.zeros(( s[0], s[1], s[2]+2*padding, s[3]+2*padding ), dtype=floatType, requires_grad=True).cuda()

    r0[:, :, padding:s[2]+1, padding:s[3]+1] = input0
    r1[:, :, padding:s[2]+1, padding:s[3]+1] = input1

    output = torch.empty((s[0], gridRadius+1, s[2], s[3]-gridRadius), dtype=floatType,  requires_grad=True).cuda()

    nEle = kernelSize**2 * s[1]

    for y in range(kernelRadius, s[2]+1, strideK):
        for x in range(kernelRadius, s[3]+1, strideK ):
            if ( x - gridRadius*strideD - kernelRadius < 0 ):
                continue

            for d in range(gridRelativeStart, 1):
                prod = r0[:, :, y-kernelRadius:y+kernelRadius+1, x-kernelRadius:x+kernelRadius+1] * \
                       r1[:, :, y-kernelRadius:y+kernelRadius+1, x+d*strideD-kernelRadius:x+d*strideD+kernelRadius+1]

                sumProd = torch.sum(prod, dim=[1,2,3])

                output[:, d+gridRadius, y - kernelRadius, x - gridRadius*strideD - kernelRadius] = \
                    sumProd / nEle

    return output

def test_backward_small( \
        padding=1, kernelSize=3, maxDisplacement=1, strideK=1, strideD=1):

    print("test_backward_small()")

    B=2
    C=2
    H=8
    W=8

    # Random tensor.
    t0 = torch.linspace(0, B*C*H*W-1, B*C*H*W, requires_grad=True).view((B, C, H, W)).float().cuda()
    t1 = t0.detach().clone()
    t1.requires_grad = True

    # Apply funcion.
    out = Corr2D_ext.forward( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    # print(out)

    gridRadius = maxDisplacement // strideD

    # Gradient.
    grad = torch.ones( (B, gridRadius+1, H, W-gridRadius) ).float().cuda()

    # Compute the autograd.
    [g0, g1] = Corr2D_ext.backward( grad, t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    # Clone new tensors.
    u0 = t0.detach().clone()
    u1 = u0.detach().clone()

    u0.requires_grad = True
    u1.requires_grad = True

    # Naive correlation.
    corr = naive_correlation( u0, u1, padding, kernelSize, maxDisplacement, strideK, strideD )

    # print(corr)

    gradU = torch.ones( (B, gridRadius+1, H, W-gridRadius) ).float().cuda()

    # Backward.
    gU = corr.backward(gradU)

    # import ipdb; ipdb.set_trace()

    print(gU) # Should be None.

    print("torch.norm(out - corr) = {}. ".format( torch.norm(out - corr) ))

    print("u0.grad.size() = {}. ".format( u0.grad.size() ))
    print("u1.grad.size() = {}. ".format( u1.grad.size() ))
    print("torch.norm(g0 - u0.grad) = {}. ".format( torch.norm(g0 - u0.grad) ))
    print("torch.norm(g1 - u1.grad) = {}. ".format( torch.norm(g1 - u1.grad) ))

def test_backward_small_double( \
        padding=1, kernelSize=3, maxDisplacement=1, strideK=1, strideD=1):

    print("test_backward_small_double()")

    B=2
    C=2
    H=8
    W=8

    # Random tensor.
    t0 = torch.linspace(0, B*C*H*W-1, B*C*H*W, dtype=torch.float64,  requires_grad=True).view((B, C, H, W)).cuda()
    t1 = t0.detach().clone()
    t1.requires_grad = True

    # Apply funcion.
    out = Corr2D_ext.forward( t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    # print(out)

    gridRadius = maxDisplacement // strideD

    # Gradient.
    grad = torch.ones( (B, gridRadius+1, H, W-gridRadius), dtype=torch.float64 ).cuda()

    # Compute the autograd.
    [g0, g1] = Corr2D_ext.backward( grad, t0, t1, \
        padding, kernelSize, maxDisplacement, strideK, strideD )

    # Clone new tensors.
    u0 = t0.detach().clone()
    u1 = u0.detach().clone()

    u0.requires_grad = True
    u1.requires_grad = True

    # Naive correlation.
    corr = naive_correlation( u0, u1, padding, kernelSize, maxDisplacement, strideK, strideD, torch.float64 )

    # print(corr)

    gradU = torch.ones( (B, gridRadius+1, H, W-gridRadius), dtype=torch.float64 ).cuda()

    # Backward.
    gU = corr.backward(gradU)

    # import ipdb; ipdb.set_trace()

    print(gU) # Should be None.

    print("torch.norm(out - corr) = {}. ".format( torch.norm(out - corr) ))

    print("u0.grad.size() = {}. ".format( u0.grad.size() ))
    print("u1.grad.size() = {}. ".format( u1.grad.size() ))
    print("torch.norm(g0 - u0.grad) = {}. ".format( torch.norm(g0 - u0.grad) ))
    print("torch.norm(g1 - u1.grad) = {}. ".format( torch.norm(g1 - u1.grad) ))

if __name__ == "__main__":
    test_backward()
    print("===")
    test_backward_small()
    print("===")
    test_backward_small_double()
