
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

    sys.path.insert(0, "/home/yaoyu/Projects/NewStereo/Trial/SmallSizeStereo/Model")
    import CommonModel as cm
else:
    from . import CommonModel as cm

import Corr2D

class FeatureExtractorParams(object):
    def __init__(self):
        super(FeatureExtractorParams, self).__init__()

        self.inChGray = 1
        self.inChGrad = 1

        self.resGrayN = 4 # 1
        self.resGrayS = 1

        self.resGradN = 1
        self.resGradS = 1

        self.preBranchN  = 2 # 2
        self.preBranchS  = 1

        # Channel number specification
        self.firstConvGrayIn   = self.inChGray
        self.firstConvGrayOut  = 32
        self.firstConvGradIn   = self.inChGrad
        self.firstConvGradOut  = 32

        self.resGrayIn  = self.firstConvGrayOut
        self.resGrayOut = self.resGrayIn

        self.resGradIn  = self.firstConvGradOut
        self.resGradOut = self.resGradIn

        # self.preBranchIn    = self.resGrayOut + self.resGradOut
        self.preBranchIn    = self.resGrayOut
        self.preBranchOut   = self.preBranchIn

        self.branch_1_In    = self.preBranchOut
        self.branch_1_Out   = 32
        self.branch_2_In    = self.preBranchOut
        self.branch_2_Out   = 32
        self.branch_3_In    = self.preBranchOut
        self.branch_3_Out   = 32
        self.concat         = self.preBranchOut + self.branch_1_Out + self.branch_2_Out + self.branch_3_Out
        self.afterBranchIn  = self.concat
        self.afterBranchOut = 128

class FeatureExtractor(nn.Module):
    def __init__(self, params):
        super(FeatureExtractor, self).__init__()

        self.params = params

        # First conv layer.
        self.firstConvGray = cm.Conv_W( self.params.firstConvGrayIn, self.params.firstConvGrayOut, 3, activation=cm.SelectedReLU() )
        self.firstConvGrad = cm.Conv_W( self.params.firstConvGradIn, self.params.firstConvGradOut, 3, activation=cm.SelectedReLU() )

        # resGray and resGrad.
        self.resGray = cm.ResPack( self.params.resGrayIn, self.params.resGrayOut, self.params.resGrayN, 3, self.params.resGrayS, \
            activation=cm.SelectedReLU(), lastActivation=cm.SelectedReLU() )
        self.resGrad = cm.ResPack( self.params.resGradIn, self.params.resGradOut, self.params.resGradN, 3, self.params.resGradS, \
            activation=cm.SelectedReLU(), lastActivation=cm.SelectedReLU() )

        # Pre-branch ResPack.
        self.preBranch = cm.ResPack( self.params.preBranchIn, self.params.preBranchOut, self.params.preBranchN, 3, self.params.preBranchS, \
            activation=cm.SelectedReLU(), lastActivation=cm.SelectedReLU() )

        # Branches.
        self.branch1 = cm.ReceptiveBranch( self.params.branch_1_In, self.params.branch_1_Out,  4, activation=cm.SelectedReLU() )
        self.branch2 = cm.ReceptiveBranch( self.params.branch_2_In, self.params.branch_2_Out,  8, activation=cm.SelectedReLU() )
        self.branch3 = cm.ReceptiveBranch( self.params.branch_3_In, self.params.branch_3_Out, 16, activation=cm.SelectedReLU() )

        # After branching conv.
        self.afterBranchConv = cm.Conv_W( self.params.afterBranchIn, self.params.afterBranchOut, 3, activation=cm.SelectedReLU() )

    def forward(self, gray, grad):
        gray = self.firstConvGray(gray)
        gray = self.resGray(gray)

        # grad = self.firstConvGrad(grad)
        # grad = self.resGrad(grad)

        # Concatenate.
        # x = torch.cat( ( gray, grad ), 1 )
        x = gray

        b0 = self.preBranch(x)
        # b0 = F.relu(b0, inplace=True)
        b1 = self.branch1(b0)
        b2 = self.branch2(b0)
        b3 = self.branch3(b0)

        hB0 = b0.size()[2]
        wB0 = b0.size()[3]

        b1 = F.interpolate( b1, ( hB0, wB0 ), mode="bilinear", align_corners=False )
        b2 = F.interpolate( b2, ( hB0, wB0 ), mode="bilinear", align_corners=False )
        b3 = F.interpolate( b3, ( hB0, wB0 ), mode="bilinear", align_corners=False )

        features = torch.cat( ( b0, b1, b2, b3 ), 1 )

        features = self.afterBranchConv(features)

        return features

class FeatureExtractor_ConvBranch(nn.Module):
    def __init__(self, params):
        super(FeatureExtractor_ConvBranch, self).__init__()

        self.params = params

        # First conv layer.
        self.firstConvGray = cm.Conv_W( self.params.firstConvGrayIn, self.params.firstConvGrayOut, 3, activation=nn.Tanh() )
        self.firstConvGrad = cm.Conv_W( self.params.firstConvGradIn, self.params.firstConvGradOut, 3, activation=nn.Tanh() )

        # resGray and resGrad.
        self.resGray = cm.ResPack( self.params.resGrayIn, self.params.resGrayOut, self.params.resGrayN, 3, self.params.resGrayS, \
            activation=nn.Tanh(), lastActivation=nn.Tanh() )
        self.resGrad = cm.ResPack( self.params.resGradIn, self.params.resGradOut, self.params.resGradN, 3, self.params.resGradS, \
            activation=nn.Tanh(), lastActivation=nn.Tanh() )

        # Pre-branch ResPack.
        self.preBranch = cm.ResPack( self.params.preBranchIn, self.params.preBranchOut, self.params.preBranchN, 3, self.params.preBranchS, \
            activation=nn.Tanh(), lastActivation=nn.Tanh() )

        # Branches.
        self.branch1 = cm.ConvBranch( self.params.branch_1_In, self.params.branch_1_Out, 1,  4, activation=nn.Tanh() )
        self.branch2 = cm.ConvBranch( self.params.branch_2_In, self.params.branch_2_Out, 2,  4, activation=nn.Tanh() )
        self.branch3 = cm.ConvBranch( self.params.branch_3_In, self.params.branch_3_Out, 3,  4, activation=nn.Tanh() )

        # After branching conv.
        self.afterBranchConv = cm.Conv_W( self.params.afterBranchIn, self.params.afterBranchOut, 3, activation=nn.Tanh() )

    def forward(self, gray, grad):
        gray = self.firstConvGray(gray)
        gray = self.resGray(gray)

        # grad = self.firstConvGrad(grad)
        # grad = self.resGrad(grad)

        # Concatenate.
        # x = torch.cat( ( gray, grad ), 1 )
        x = gray

        b0 = self.preBranch(x)
        # b0 = F.relu(b0, inplace=True)
        b1 = self.branch1(b0)
        b2 = self.branch2(b0)
        b3 = self.branch3(b0)

        hB0 = b0.size()[2]
        wB0 = b0.size()[3]

        b1 = F.interpolate( b1, ( hB0, wB0 ), mode="bilinear", align_corners=False )
        b2 = F.interpolate( b2, ( hB0, wB0 ), mode="bilinear", align_corners=False )
        b3 = F.interpolate( b3, ( hB0, wB0 ), mode="bilinear", align_corners=False )

        features = torch.cat( ( b0, b1, b2, b3 ), 1 )

        features = self.afterBranchConv(features)

        return features

class DisparityRegression(nn.Module):
    def __init__(self, maxDisp, flagCPU=False):
        super(DisparityRegression, self).__init__()
        
        dispSequence = np.reshape( np.array( range(maxDisp, -1, -1) ), [1, maxDisp + 1, 1, 1] )

        self.disp = torch.from_numpy( dispSequence ).float()
        self.disp.requires_grad = False

        if ( not flagCPU ):
            self.disp = self.disp.cuda()

    def forward(self, x):
        disp = self.disp.repeat( x.size()[0], 1, x.size()[2], x.size()[3] )
        out  = torch.sum( x * disp, 1 ).unsqueeze(1)
        
        return out

class EDRegression(nn.Module):
    def __init__(self, inCh):
        super(EDRegression, self).__init__()

        self.encoder0_In  = inCh
        self.encoder0_Out = 256
        self.encoder1_In  = self.encoder0_Out
        self.encoder1_Out = self.encoder1_In
        self.encoder2_In  = self.encoder1_Out
        self.encoder2_Out = self.encoder2_In
        
        self.decoder3_In  = self.encoder2_Out
        self.decoder3_Out = self.decoder3_In // 2
        self.decoder2_In  = self.decoder3_Out
        self.decoder2_Out = self.decoder3_Out
        self.decoder1_In  = self.decoder2_Out + self.encoder1_Out
        self.decoder1_Out = self.decoder3_Out
        self.decoder0_In  = self.decoder1_Out + self.encoder0_Out
        self.decoder0_Out = 128

        # Encoder-decoder.
        self.e0 = cm.Conv_Half( self.encoder0_In, self.encoder0_Out, k=3, activation=cm.SelectedReLU() )
        self.e1 = cm.Conv_Half( self.encoder1_In, self.encoder1_Out, k=3, activation=cm.SelectedReLU() )
        self.e2 = cm.Conv_Half( self.encoder2_In, self.encoder2_Out, k=3, activation=cm.SelectedReLU() )

        self.d3 = cm.Conv_W( self.decoder3_In, self.decoder3_Out, k=3, activation=cm.SelectedReLU() )
        self.d2 = cm.Deconv_DoubleSize( self.decoder2_In, self.decoder2_Out, op=1, activation=cm.SelectedReLU() )
        self.d1 = cm.Deconv_DoubleSize( self.decoder1_In, self.decoder1_Out, op=1, activation=cm.SelectedReLU() )
        self.d0 = cm.Deconv_DoubleSize( self.decoder0_In, self.decoder0_Out, op=1, activation=cm.SelectedReLU() )

        # Regression.
        self.bp  = cm.Conv_W( self.encoder0_In, self.decoder0_Out, k=3, activation=cm.SelectedReLU() )
        self.rg0 = cm.Conv_W( self.decoder0_Out, 64, k=3, activation=cm.SelectedReLU() )
        self.rg1 = cm.Conv_W( 64, 1, k=3 )

    def forward(self, x):
        fe0 = self.e0(x)
        fe1 = self.e1(fe0)
        fe2 = self.e2(fe1)

        # import ipdb; ipdb.set_trace()

        fd3 = self.d3(fe2)
        fd2 = self.d2(fd3)
        fd2 = torch.cat( (fd2, fe1), 1 )
        fd1 = self.d1(fd2)
        fd1 = torch.cat( (fd1, fe0), 1 )
        fd0 = self.d0(fd1)

        # By-pass.
        bp = self.bp(x)

        # Regression.
        disp0 = self.rg0( fd0 + bp.mul(0.1) )
        disp1 = self.rg1( disp0 )

        return disp1

class SmallSizeStereoParams(object):
    def __init__(self):
        super(SmallSizeStereoParams, self).__init__()

        self.featureExtractorParams = FeatureExtractorParams()

        self.maxDisp = 64

        # Correlation.
        self.corrPadding    = 1
        self.corrKernelSize = 3
        self.corrStrideK    = 1
        self.corrStrideD    = 1

class SmallSizeStereo(nn.Module):
    def __init__(self, params):
        super(SmallSizeStereo, self).__init__()

        self.params = params

        # Feature extractors.
        self.fe = FeatureExtractor_ConvBranch(self.params.featureExtractorParams)

        # Correlation.
        self.corr2dm = Corr2D.Corr2DM( self.params.maxDisp, \
            padding=self.params.corrPadding, \
            kernelSize=self.params.corrKernelSize, \
            strideK=self.params.corrStrideK, \
            strideD=self.params.corrStrideD )

        # # Disparity regression.
        # self.dr = DisparityRegression(self.params.maxDisp, False)
        self.edr = EDRegression( self.params.maxDisp + 1 )

        # Initialization.
        for m in self.modules():
            # print(m)
            if ( isinstance( m, (nn.Conv2d) ) ):
                n = m.kernel_size[0] * m.kernel_size[1]
                # m.weight.data.normal_(0, math.sqrt( 2.0 / n )
                m.weight.data.normal_(1/n, math.sqrt( 2.0 / n ))
                m.weight.data = m.weight.data / m.in_channels
            elif ( isinstance( m, (nn.Conv3d) ) ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt( 2.0 / n ))
                m.weight.data.uniform_(0, math.sqrt( 2.0 / n ))
            elif ( isinstance( m, (nn.BatchNorm2d) ) ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif ( isinstance( m, (nn.BatchNorm3d) ) ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif ( isinstance( m, (nn.Linear) ) ):
                m.weight.data.uniform_(0, 1)
                m.bias.data.zero_()
            # else:
            #     raise Exception("Unexpected module type {}.".format(type(m)))

    def forward(self, gray0, gray1, grad0, grad1):
        # Feature extraction.
        f0 = self.fe(gray0, grad0)
        f1 = self.fe(gray1, grad1)

        # Correlation.
        cost = self.corr2dm(f0, f1)

        # Disparity.
        # disp = F.softmax( cost, dim = 1 )
        # disp = self.dr(disp)

        disp = self.edr( cost )

        # import ipdb; ipdb.set_trace()

        return disp

class SmallSizeStereo_NoCorr(nn.Module):
    def __init__(self, params):
        super(SmallSizeStereo_NoCorr, self).__init__()

        self.params = params

        # Feature extractors.
        self.fe = FeatureExtractor(self.params.featureExtractorParams)

        # After combining the two feature maps.
        self.combineConv = cm.Conv_W( \
            self.params.featureExtractorParams.afterBranchOut*2, \
            self.params.featureExtractorParams.afterBranchOut*2, k=3, activation=cm.SelectedReLU() )
        
        self.encoder0_In  = self.params.featureExtractorParams.afterBranchOut*2
        self.encoder0_Out = 256
        self.encoder1_In  = self.encoder0_Out
        self.encoder1_Out = self.encoder1_In
        self.encoder2_In  = self.encoder1_Out
        self.encoder2_Out = self.encoder2_In
        
        self.decoder3_In  = self.encoder2_Out
        self.decoder3_Out = self.decoder3_In // 2
        self.decoder2_In  = self.decoder3_Out
        self.decoder2_Out = self.decoder3_Out
        self.decoder1_In  = self.decoder2_Out + self.encoder1_Out
        self.decoder1_Out = self.decoder3_Out
        self.decoder0_In  = self.decoder1_Out + self.encoder0_Out
        self.decoder0_Out = 128

        # Encoder-decoder.
        self.e0 = cm.Conv_Half( self.encoder0_In, self.encoder0_Out, k=3, activation=cm.SelectedReLU() )
        self.e1 = cm.Conv_Half( self.encoder1_In, self.encoder1_Out, k=3, activation=cm.SelectedReLU() )
        self.e2 = cm.Conv_Half( self.encoder2_In, self.encoder2_Out, k=3, activation=cm.SelectedReLU() )

        self.d3 = cm.Conv_W( self.decoder3_In, self.decoder3_Out, k=3, activation=cm.SelectedReLU() )
        self.d2 = cm.Deconv_DoubleSize( self.decoder2_In, self.decoder2_Out, op=1, activation=cm.SelectedReLU() )
        self.d1 = cm.Deconv_DoubleSize( self.decoder1_In, self.decoder1_Out, op=1, activation=cm.SelectedReLU() )
        self.d0 = cm.Deconv_DoubleSize( self.decoder0_In, self.decoder0_Out, op=1, activation=cm.SelectedReLU() )

        # Regression.
        self.bp  = cm.Conv_W( self.encoder0_In, self.decoder0_Out, k=3, activation=cm.SelectedReLU() )
        self.rg0 = cm.Conv_W( self.decoder0_Out, 64, k=3, activation=cm.SelectedReLU() )
        self.rg1 = cm.Conv_W( 64, 1, k=3 )

        # Initialization.
        for m in self.modules():
            # print(m)
            if ( isinstance( m, (nn.Conv2d) ) ):
                n = m.kernel_size[0] * m.kernel_size[1]
                # m.weight.data.normal_(0, math.sqrt( 2.0 / n )
                m.weight.data.normal_(1/n, math.sqrt( 2.0 / n ))
                m.weight.data = m.weight.data / m.in_channels
            elif ( isinstance( m, (nn.Conv3d) ) ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt( 2.0 / n ))
                m.weight.data.uniform_(0, math.sqrt( 2.0 / n ))
            elif ( isinstance( m, (nn.BatchNorm2d) ) ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif ( isinstance( m, (nn.BatchNorm3d) ) ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif ( isinstance( m, (nn.Linear) ) ):
                m.weight.data.uniform_(0, 1)
                m.bias.data.zero_()
            # else:
            #     raise Exception("Unexpected module type {}.".format(type(m)))

    def forward(self, gray0, gray1, grad0, grad1):
        # Feature extraction.
        f0 = self.fe(gray0, grad0)
        f1 = self.fe(gray1, grad1)

        # Concatenate.
        f = torch.cat( ( f0, f1 ), 1 )
        f = self.combineConv(f)

        fe0 = self.e0(f)
        fe1 = self.e1(fe0)
        fe2 = self.e2(fe1)

        # import ipdb; ipdb.set_trace()

        fd3 = self.d3(fe2)
        fd2 = self.d2(fd3)
        fd2 = torch.cat( (fd2, fe1), 1 )
        fd1 = self.d1(fd2)
        fd1 = torch.cat( (fd1, fe0), 1 )
        fd0 = self.d0(fd1)

        # By-pass.
        bp = self.bp(f)

        # Regression.
        disp0 = self.rg0( fd0 + bp.mul(0.1) )
        disp1 = self.rg1( disp0 )

        return disp1[:, :, :, self.params.maxDisp:]

if __name__ == "__main__":
    print("Test SmallSizeStereo.py")

    params = SmallSizeStereo()

    sss = SmallSizeStereo(params)

    print("sss has %d model parameters. " % ( \
        sum( [ p.data.nelement() for p in sss.parameters() ] ) ))

    modelDict = sss.state_dict()
    for item in modelDict:
        print("Layer {}. ".format(item))
