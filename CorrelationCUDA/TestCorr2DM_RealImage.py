from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch

import Corr2D

def test_gradcheck(B=2, C=2, H=8, W=8, \
        padding=1, kernelSize=3, maxDisplacement=4, strideK=1, strideD=1):

    print("test_backward()")

    # Random tensor.
    t0 = torch.autograd.Variable( torch.rand((B, C, H, W)).double().cuda(), requires_grad=True )
    t1 = t0.clone().detach()

    # Create a Corr2DM object.
    corr2d = Corr2D.Corr2DM( maxDisplacement, padding=padding, kernelSize=kernelSize, strideK=strideK, strideD=strideD )

    # Check gradient.
    test = torch.autograd.gradcheck( corr2d, ( t0, t1 ), eps=1e-6, atol=1e-4 )
    print(test)

def test_two_tensors(t0, t1, \
    padding=1, kernelSize=3, maxDisplacement=64, strideK=1, strideD=1):
    
    # Create a Corr2DM object.
    corr2d = Corr2D.Corr2DM( maxDisplacement, padding=padding, kernelSize=kernelSize, strideK=strideK, strideD=strideD )

    out = corr2d( t0, t1 )

    return out

def zero_normalized_cross_correlation(x0, x1):
    x0 = x0.reshape((1,-1))
    x1 = x1.reshape((-1,1))

    n0 = np.linalg.norm(x0)
    n1 = np.linalg.norm(x1)

    nx0 = x0 / n0
    nx1 = x1 / n1

    return nx0.dot(nx1)

if __name__ == "__main__":
    # torch.autograd.gradcheck()
    # test_gradcheck()

    # # Load two images.
    # img0 = cv2.imread("/home/yaoyu/temp/SceneFlowSample/FlyingThings3D/RGB_cleanpass/left/0006.png", cv2.IMREAD_UNCHANGED)
    # img1 = cv2.imread("/home/yaoyu/temp/SceneFlowSample/FlyingThings3D/RGB_cleanpass/right/0006.png", cv2.IMREAD_UNCHANGED)

    # # Convert the images into tensors.
    # t0 = torch.from_numpy(img0).double().permute((2, 0, 1)).unsqueeze(0).cuda()
    # t1 = torch.from_numpy(img1).double().permute((2, 0, 1)).unsqueeze(0).cuda()

    a = np.array( range(256, 0, -1) )
    b = np.stack( (a, a, a), axis=0 )

    b_0 = b[0:3, 0:3]
    b_1 = b[0:3, 63:66]

    print("b[0:3, 0:3] = \n{}".format( b_0 ))
    print("b[0:3, 63:66] = \n{}".format( b_1 ))

    zncc = zero_normalized_cross_correlation(b_0, b_1)
    print("zncc = \n{}".format(zncc))

    t0 = torch.from_numpy( b ).double().unsqueeze(0).unsqueeze(0)
    t1 = t0.clone()

    t0 = t0.cuda()
    t1 = t1.cuda()

    # Set the requires_grad flag.
    t0.requires_grad = True
    t1.requires_grad = True

    # Test two indentical tensors.
    cost = test_two_tensors(t0, t1)

    costCPU = cost.detach().cpu().numpy()

    # import ipdb; ipdb.set_trace()

    print(costCPU[0, :, 1, 0])

    plt.plot( costCPU[0, :, 1, 0], "-*" )
    plt.show()