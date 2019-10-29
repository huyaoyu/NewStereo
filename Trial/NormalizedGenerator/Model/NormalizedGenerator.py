
from __future__ import print_function

# Some structures of this file is inspired by 
# the research work done by Chang and Chen, 
# Pyramid Stereo Matching Network, CVPR2018.
# https://github.com/JiaRenChang/PSMNet
# 
# And
#
# The work by Bee Lim, et. al.
# https://github.com/thstkdgus35/EDSR-PyTorch
#

import cv2
import math
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

if ( __name__ == "__main__" ):
    import sys

    sys.path.insert(0, "/home/yaoyu/Projects/NewStereo/Trial/NormalizedGenerator/Model")
    import CommonModel
else:
    from . import CommonModel

class Upsampler(nn.Module):
    def __init__(self, inCh, r):
        """
        r must be a positive even number.
        """

        super(Upsampler, self).__init__()

        assert r > 0

        self.r = r
        self.kernelSize = 3

        transCh = int( r**2 * inCh )

        self.model = nn.Sequential( \
            CommonModel.Conv_W(inCh, transCh, self.kernelSize), \
            nn.PixelShuffle(upscale_factor=self.r), \
            nn.ReLU(True) )
    
    def forward(self, x):
        return self.model(x)

class EDSR2(nn.Module):
    def __init__(self, inCh, transCh, outCh, resN, resScale=1):
        super(EDSR2, self).__init__()

        self.kernelSize = 3

        entryConv = CommonModel.Conv_W( inCh, transCh, self.kernelSize )
        lastConv  = CommonModel.Conv_W( transCh, outCh, self.kernelSize )

        resPack = CommonModel.ResPack( transCh, transCh, resN, self.kernelSize, resScale )
        upsampler = Upsampler( transCh, 2 )

        self.model = nn.Sequential( \
            entryConv, \
            resPack, \
            upsampler, \
            lastConv )

    def forward(self, x):
        return self.model(x)

class NormalizedGeneratorParams(object):
    def __init__(self):
        super(NormalizedGeneratorParams, self).__init__()
    
        self.inCh        = 2
        self.preBranchN  = 2
        self.EDSR2N      = 2
        self.EDSR2S      = 1
        self.fsrResPackN = 2
        self.fsrResPackS = 1 # Scale.

        # Channel number specification
        self.firstConvIn    = self.inCh
        self.firstConvOut   = 16
        self.preBranchIn    = self.firstConvOut
        self.preBranchOut   = 16
        self.branch4In      = self.preBranchOut
        self.branch4Out     = 16
        self.branch16In     = self.preBranchOut
        self.branch16Out    = 16
        self.branch64In     = self.preBranchOut
        self.branch64Out    = 16
        self.concat         = self.preBranchOut + self.branch4Out + self.branch16Out + self.branch64Out
        self.afterBranchIn  = self.concat
        self.afterBranchOut = 16
        self.EDSR2In        = self.afterBranchOut
        self.EDSR2Trans     = self.EDSR2In
        self.EDSR2Out       = 1
        self.fsrConv1In     = 3
        self.fsrConv1Out    = 16
        self.fsrResPackIn   = self.fsrConv1Out
        self.fsrResPackOut  = 16
        self.fsrConv2In     = self.fsrResPackOut
        self.fsrConv2Out    = 1

class NormalizedGenerator(nn.Module):
    def __init__(self, params):
        super(NormalizedGenerator, self).__init__()

        self.flagCPU = False

        self.params = params

        # BatchNorm for the images.
        self.bn = nn.BatchNorm2d(1, track_running_stats=False)

        # First conv layer.
        self.firstConv = CommonModel.Conv_W( self.params.firstConvIn, self.params.firstConvOut, 3 )

        # Pre-branch ResPack.
        self.preBranch = CommonModel.ResPack( self.params.preBranchIn, self.params.preBranchOut, self.params.preBranchN, 3  )

        # Branches.
        self.branch4  = CommonModel.ReceptiveBranch( self.params.branch4In,  self.params.branch4Out,   4 )
        self.branch16 = CommonModel.ReceptiveBranch( self.params.branch16In, self.params.branch16Out, 16 )
        self.branch64 = CommonModel.ReceptiveBranch( self.params.branch64In, self.params.branch64Out, 64 )

        # After branching conv.
        self.afterBranchConv = CommonModel.Conv_W( self.params.afterBranchIn, self.params.afterBranchOut, 3 )

        # EDSR2.
        self.edsr2 = EDSR2( self.params.EDSR2In, self.params.EDSR2Trans, self.params.EDSR2Out, \
            self.params.EDSR2N, self.params.EDSR2S )
        
        # Full size refinement conv.
        self.fsrConv1 = CommonModel.Conv_W( self.params.fsrConv1In, self.params.fsrConv1Out, 3 )

        # Full size refinement ResPack.
        self.fsrResPack = CommonModel.ResPack( self.params.fsrResPackIn, self.params.fsrResPackOut, \
            self.params.fsrResPackN, 3, self.params.fsrResPackS )

        # Full size refinement last conv.
        self.fsrConv2 = CommonModel.Conv_W( self.params.fsrConv2In, self.params.fsrConv2Out, 3 )

        # Initialization.
        for m in self.modules():
            if ( isinstance( m, (nn.Conv2d) ) ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt( 2.0 / n ))
            elif ( isinstance( m, (nn.Conv3d) ) ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt( 2.0 / n ))
            elif ( isinstance( m, (nn.BatchNorm2d) ) ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif ( isinstance( m, (nn.BatchNorm3d) ) ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif ( isinstance( m, (nn.Linear) ) ):
                m.bias.data.zero_()
            # else:
            #     raise PyramidNetException("Unexpected module type {}.".format(type(m)))
    
    def set_cpu_mode(self):
        self.flagCPU = True

    def unset_cpu_mode(self):
        self.flagCPU = False

    def forward(self, dispLH, imgLH, imgL):
        imgLH = self.bn(imgLH)
        imgL  = self.bn(imgL)
        
        # Concat dispLH and imgLH.
        x = torch.cat( ( dispLH, imgLH ), 1 )

        # Denoising.
        x   = self.firstConv(x)
        b1  = self.preBranch(x)
        b4  = self.branch4(b1)
        b16 = self.branch16(b1)
        b64 = self.branch64(b1)

        hB1 = b1.size()[2]
        wB1 = b1.size()[3]

        b4  = F.interpolate( b4,  ( hB1, wB1 ), mode="bilinear", align_corners=False )
        b16 = F.interpolate( b16, ( hB1, wB1 ), mode="bilinear", align_corners=False )
        b64 = F.interpolate( b64, ( hB1, wB1 ), mode="bilinear", align_corners=False )

        features = torch.cat( ( b1, b4, b16, b64 ), 1 )

        features = self.afterBranchConv(features)

        fsDisp = self.edsr2(features)

        # Pre-pare the additional input for full size refinement.
        diff = imgL - F.interpolate(imgLH, ( imgL.size()[2], imgL.size()[3] ), mode="bilinear", align_corners=False)
        
        # Full size refinement.
        x = torch.cat( (fsDisp, imgL, diff), 1 )

        x = self.fsrConv1(x)
        x = self.fsrResPack(x)
        rFsDisp = self.fsrConv2(x)

        return fsDisp, rFsDisp

if __name__ == "__main__":
    print("Test NormalizedGenerator.py")

    params = NormalizedGeneratorParams()

    ng = NormalizedGenerator(params)

    print("ng has %d model parameters. " % ( \
        sum( [ p.data.nelement() for p in ng.parameters() ] ) ))

    modelDict = ng.state_dict()
    for item in modelDict:
        print("Layer {}. ".format(item))