
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

    sys.path.insert(0, "/home/yaoyu/Projects/NewStereo/Trial/PRU/Model")
    import CommonModel as cm
    from StereoUtility import WarpByDisparity
else:
    from . import CommonModel as cm
    from .StereoUtility import WarpByDisparity

import Corr2D

class ConvExtractor(nn.Module):
    def __init__(self, inCh, outCh, lastActivation=None):
        super(ConvExtractor, self).__init__()

        moduleList = [ \
            cm.Conv_Half( inCh, outCh, activation=nn.LeakyReLU(0.1) ), \
            cm.Conv_W( outCh, outCh, activation=nn.LeakyReLU(0.1) ), \
            cm.Conv_W( outCh, outCh, activation=nn.LeakyReLU(0.1) ) ]
        
        if ( lastActivation is not None ):
            moduleList.append( lastActivation )

        self.model = nn.Sequential( *moduleList )

    def forward(self, x):
        return self.model(x)

class FeatureNormalization(nn.Module):
    def __init__(self, inCh):
        super(FeatureNormalization, self).__init__()

        assert inCh > 0

        # if ( cm.is_odd(inCh) ):
        #     raise Exception("inCh = {} which is not an even number.".format(inCh))

        self.model = nn.BatchNorm2d(inCh, track_running_stats=False)

    def forward(self, x):
        return self.model(x)

class CostRegulator(nn.Module):
    def __init__(self, inCh, interChList, lastActivation=None):
        super(CostRegulator, self).__init__()

        n = len(interChList)

        if ( 0 == n ):
            raise Exception("n = {}. ".format(n))

        cumCh = np.cumsum( interChList ).tolist()
        cumCh = [ 0, *cumCh ]

        self.regulators = nn.ModuleList()

        for i in range(n):
            self.regulators.append( cm.Conv_W( inCh + cumCh[i], interChList[i], activation=nn.LeakyReLU(0.1) ) )

        self.lastActivation = lastActivation
    
    def forward(self, x):
        n = len(self.regulators)

        for i in range(n):
            x = torch.cat( ( self.regulators[i](x), x ), 1 )

        if ( self.lastActivation is not None ):
            x = self.lastActivation(x)
        
        return x

class PredictDisparity(nn.Module):
    def __init__(self, inCh):
        super(PredictDisparity, self).__init__()

        self.model = nn.Sequential( \
            cm.Conv_W(inCh, 64, activation=nn.LeakyReLU(0.1)), \
            cm.Conv_W(64, 1) )
        
        # self.model = \
        #     cm.Conv_W(inCh, 1, activation=nn.LeakyReLU(0.1))

    def forward(self, x):
        return self.model(x)

class UpDisparity(nn.Module):
    def __init__(self):
        super(UpDisparity, self).__init__()

        self.model = cm.Deconv_DoubleSize(1, 1, k=4, p=1)

    def forward(self, x):
        return self.model(x)

class UpFeature(nn.Module):
    def __init__(self, inCh, outCh):
        super(UpFeature, self).__init__()

        self.model = cm.Deconv_DoubleSize(inCh, outCh, k=4, p=1)

    def forward(self, x):
        return self.model(x)

class Cost2DisparityAndFeature(nn.Module):
    def __init__(self, inCh, outFeatCh, interChList, flagUp=True):
        super(Cost2DisparityAndFeature, self).__init__()

        # The regulator.
        self.regulator = CostRegulator(inCh, interChList)

        cumCh = np.cumsum( interChList )

        # The disparity predictor.
        self.disp = PredictDisparity( inCh + cumCh[-1] )

        # The up-sample model.
        if ( flagUp ):
            self.upDisp = UpDisparity()
            self.upFeat = UpFeature( inCh + cumCh[-1], outFeatCh )
        else:
            self.upDisp = None
            self.upFeat = None

    def forward(self, x):
        x = self.regulator(x)

        disp = self.disp(x)

        if ( self.upDisp is not None and self.upFeat is not None ):
            upDisp = self.upDisp(disp)
            upFeat = self.upFeat(x)

            return disp, upDisp, upFeat
        else:
            return disp, x

class Cost2DisparityAndFeatureRes(nn.Module):
    def __init__(self, inCh, interChList, flagUp=True):
        super(Cost2DisparityAndFeatureRes, self).__init__()

        # The regulator.
        self.regulator = CostRegulator(inCh, interChList)

        cumCh = np.cumsum( interChList )

        # The disparity predictor.
        self.disp = PredictDisparity( inCh + cumCh[-1] )

        # The up-sample model.
        if ( flagUp ):
            self.upDisp = UpDisparity()
        else:
            self.upDisp = None

    def forward(self, x, lastDisp):
        x = self.regulator(x)

        dispRes = self.disp(x)
        disp    = dispRes + lastDisp

        if ( self.upDisp is not None ):
            upDisp = self.upDisp(disp)

            return disp, upDisp, dispRes
        else:
            return disp, x, dispRes

class DisparityRefine(nn.Module):
    def __init__(self, inCh):
        super(DisparityRefine, self).__init__()

        self.model = nn.Sequential( \
            nn.Conv2d(inCh, 128, kernel_size=3, stride=1, padding=1,  dilation=1,  bias=True), nn.LeakyReLU(0.1), \
            nn.Conv2d(128,  128, kernel_size=3, stride=1, padding=2,  dilation=2,  bias=True), nn.LeakyReLU(0.1), \
            nn.Conv2d(128,  128, kernel_size=3, stride=1, padding=4,  dilation=4,  bias=True), nn.LeakyReLU(0.1), \
            nn.Conv2d(128,  96,  kernel_size=3, stride=1, padding=8,  dilation=8,  bias=True), nn.LeakyReLU(0.1), \
            nn.Conv2d(96,   64,  kernel_size=3, stride=1, padding=16, dilation=16, bias=True), nn.LeakyReLU(0.1), \
            nn.Conv2d(64,   64,  kernel_size=3, stride=1, padding=1,  dilation=1,  bias=True), nn.LeakyReLU(0.1), \
            PredictDisparity(64) \
             )

    def forward(self, disp, fe):
        return self.model(fe) + disp

class EDRegression(nn.Module):
    def __init__(self, inCh):
        super(EDRegression, self).__init__()

        self.dispNorm = nn.BatchNorm2d(1, track_running_stats=False)
        self.dispFE   = nn.Sequential( \
            cm.Conv_W( 1, inCh, k=3, activation=None ), \
            cm.ResBlock( inCh, k=3, activation=cm.SelectedReLU() ), \
            nn.BatchNorm2d( inCh, track_running_stats=False ) )

        self.featNorm = nn.BatchNorm2d(inCh, track_running_stats=False)

        self.encoder0_In  = inCh*2
        self.encoder0_Out = 64
        self.encoder1_In  = self.encoder0_Out
        self.encoder1_Out = self.encoder1_In * 2
        self.encoder2_In  = self.encoder1_Out
        self.encoder2_Out = self.encoder2_In
        
        self.decoder3_In  = self.encoder2_Out
        self.decoder3_Out = self.decoder3_In // 2
        self.decoder2_In  = self.decoder3_Out
        self.decoder2_Out = self.decoder3_Out
        self.decoder1_In  = self.decoder2_Out + self.encoder1_Out
        self.decoder1_Out = self.decoder3_Out
        self.decoder0_In  = self.decoder1_Out + self.encoder0_Out
        self.decoder0_Out = 64

        # Encoder-decoder.
        self.e0 = cm.Conv_Half( self.encoder0_In, self.encoder0_Out, k=3, activation=cm.SelectedReLU() )
        self.e1 = cm.Conv_Half( self.encoder1_In, self.encoder1_Out, k=3, activation=cm.SelectedReLU() )
        self.e2 = cm.Conv_Half( self.encoder2_In, self.encoder2_Out, k=3, activation=cm.SelectedReLU() )

        self.d3 = cm.Conv_W( self.decoder3_In, self.decoder3_Out, k=3, activation=cm.SelectedReLU() )
        self.d2 = cm.Deconv_DoubleSize( self.decoder2_In, self.decoder2_Out, k=4, p=1, activation=cm.SelectedReLU() )
        self.d1 = cm.Deconv_DoubleSize( self.decoder1_In, self.decoder1_Out, k=4, p=1, activation=cm.SelectedReLU() )
        self.d0 = cm.Deconv_DoubleSize( self.decoder0_In, self.decoder0_Out, k=4, p=1, activation=cm.SelectedReLU() )

        # self.finalUp = cm.Deconv_DoubleSize( self.decoder0_Out, 64, k=4, p=1, activation=cm.SelectedReLU() )

        # Regression.
        # self.bp  = cm.Conv_W( self.encoder0_In, self.decoder0_Out, k=3, activation=cm.SelectedReLU() )
        
        self.rg0 = cm.Conv_W( self.decoder0_Out, 64, k=3, activation=cm.SelectedReLU() )
        self.rg1 = cm.Conv_W( 64, 1, k=3 )

    def forward(self, x, disp):
        disp = self.dispNorm(disp)
        dispFeat = self.dispFE(disp)

        x = self.featNorm(x)

        c = torch.cat( (dispFeat, x), 1 )

        fe0 = self.e0(c)
        fe1 = self.e1(fe0)
        fe2 = self.e2(fe1)

        # import ipdb; ipdb.set_trace()

        fd3 = self.d3(fe2)
        fd2 = self.d2(fd3)
        fd2 = torch.cat( (fd2, fe1), 1 )
        fd1 = self.d1(fd2)
        fd1 = torch.cat( (fd1, fe0), 1 )
        fd0 = self.d0(fd1)
        # fd0 = self.finalUp(fd0)

        # # By-pass.
        # bp = self.bp(x)

        # Regression.
        disp0 = self.rg0( fd0 )
        disp1 = self.rg1( disp0 )

        return disp1

class PWCNetStereoParams(object):
    def __init__(self):
        super(PWCNetStereoParams, self).__init__()

        self.flagGray = True
        self.inCh     = 65

        self.amp = 1

        self.maxDisp = 16

        # Correlation.
        self.corrPadding    = self.maxDisp
        self.corrKernelSize = 1
        self.corrStrideK    = 1
        self.corrStrideD    = 1

    def set_max_disparity(self, md):
        assert md > 0

        self.maxDisp = int( md )
        self.corrPadding = self.maxDisp

class PWCNetStereoRes(nn.Module):
    def __init__(self, params):
        super(PWCNetStereoRes, self).__init__()

        self.params = params

        # Feature extractors.
        if ( self.params.flagGray ):
            self.fe1 = ConvExtractor( self.params.inCh, 32)
            self.re1 = nn.Sequential( \
                ConvExtractor( self.params.inCh, 32), \
                cm.Deconv_DoubleSize( 32, 32, k=4, p=1, activation=cm.SelectedReLU() ) )
        else:
            raise Exception("Only supports grayscale at the moment.")

            # self.fe1 = ConvExtractor( 3, 32)
            # self.re1 = nn.Sequential( \
            #     ConvExtractor( 3, 32), \
            #     cm.Deconv_DoubleSize( 32, 32, k=4, p=1, activation=cm.SelectedReLU() ) )
        
        self.fe2 = ConvExtractor( 32,  64)
        self.fe3 = ConvExtractor( 64,  128)
        # self.fe4 = ConvExtractor( 64,  128)

        # Batch normalization layers.
        self.fn1 = FeatureNormalization(32)
        self.fn2 = FeatureNormalization(64)
        self.fn3 = FeatureNormalization(128)
        # self.fn4 = FeatureNormalization(128)

        # Correlation.
        self.corr2dm = Corr2D.Corr2DM( self.params.maxDisp, \
            padding=self.params.corrPadding, \
            kernelSize=self.params.corrKernelSize, \
            strideK=self.params.corrStrideK, \
            strideD=self.params.corrStrideD )

        nd = self.params.maxDisp + 1 + self.params.maxDisp

        self.corrActivation = cm.Conv_W(nd, nd, k=1)
        self.costFeatureExtractor3 = nn.Sequential( \
            cm.Conv_W(128, nd, k=3, activation=None), \
            FeatureNormalization(nd) )
        self.costFeatureExtractor2 = nn.Sequential( \
            cm.Conv_W(64, nd, k=3, activation=None), \
            FeatureNormalization(nd) )
        self.costFeatureExtractor1 = nn.Sequential( \
            cm.Conv_W(32, nd, k=3, activation=None), \
            FeatureNormalization(nd) )

        # Disparity at various scale.
        # interChList = [ 128, 128, 96, 64, 32 ]
        interChList = [ 96, 64, 32 ]
        chFeat      = np.sum( interChList )

        # self.disp5 = Cost2DisparityAndFeature(nd, 1, interChList)
        # self.disp4 = Cost2DisparityAndFeatureRes(nd + 128, interChList)
        # self.disp3 = Cost2DisparityAndFeatureRes(nd + 64, interChList)
        
        # self.disp3 = Cost2DisparityAndFeature(nd+128, 1, interChList)
        # self.disp2 = Cost2DisparityAndFeatureRes(nd + 64, interChList)
        # self.disp1 = Cost2DisparityAndFeatureRes(nd + 32, interChList, flagUp=False)

        self.disp3 = Cost2DisparityAndFeature(nd + nd, 1, interChList)
        self.disp2 = Cost2DisparityAndFeatureRes(nd + nd, interChList)
        self.disp1 = Cost2DisparityAndFeatureRes(nd + nd, interChList, flagUp=True)

        self.refine = EDRegression( 32 )

        # Warp.
        self.warp = WarpByDisparity()

        # Initialization.
        # for m in self.modules():
        #     # print(m)
        #     if ( isinstance( m, (nn.Conv2d) ) ):
        #         n = m.kernel_size[0] * m.kernel_size[1]
        #         # m.weight.data.normal_(0, math.sqrt( 2.0 / n )
        #         m.weight.data.normal_(1/n, math.sqrt( 2.0 / n ))
        #         m.weight.data = m.weight.data / m.in_channels
        #     elif ( isinstance( m, (nn.Conv3d) ) ):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        #         # m.weight.data.normal_(0, math.sqrt( 2.0 / n ))
        #         m.weight.data.uniform_(0, math.sqrt( 2.0 / n ))
        #     elif ( isinstance( m, (nn.BatchNorm2d) ) ):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif ( isinstance( m, (nn.BatchNorm3d) ) ):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif ( isinstance( m, (nn.Linear) ) ):
        #         m.weight.data.uniform_(0, 1)
        #         m.bias.data.zero_()
        #     # else:
        #     #     raise Exception("Unexpected module type {}.".format(type(m)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, gray0, gray1, grad0, grad1):
        B, C, H, W = gray0.size()
        
        # Feature extraction.
        f10 = self.fe1(gray0)
        f11 = self.fe1(gray1)
        
        f20 = self.fe2(f10)
        f21 = self.fe2(f11)

        f30 = self.fe3(f20)
        f31 = self.fe3(f21)

        # f40 = self.fe4(f30)
        # f41 = self.fe4(f31)

        # # ========== Scale 4. ==========
        # scale = 4

        # # Warp.
        # # warp4 = self.warp( f41, upDisp5 * self.params.amp * 0.5**scale )
        # upDisp5 = upDisp5 * ( 2 )
        # warp4 = self.warp( f41, upDisp5 / self.params.amp )

        # # Normalization
        # f40   = self.fn4(f40)
        # warp4 = self.fn4(warp4)

        # # Correlation.
        # cost4 = self.corr2dm( f40, warp4 )
        # cost4 = self.corrActivation( cost4 )

        # # Concatenate.
        # cost4 = torch.cat( (cost4, f40), 1 )

        # # Disparity.
        # disp4, upDisp4 = self.disp4(cost4, upDisp5)

        # ========== Scale 3. ==========
        scale = 3

        # Warp.
        # warp3 = self.warp( f31, upDisp4 * self.params.amp * 0.5**scale )
        # upDisp4 = upDisp4 * ( 2 )
        # warp3 = self.warp( f31, upDisp4 / self.params.amp )

        warp3 = f31

        # Normalization
        f30   = self.fn3(f30)
        warp3 = self.fn3(warp3)

        # Correlation.
        cost3 = self.corr2dm( f30, warp3 )
        cost3 = self.corrActivation( cost3 )

        # Concatenate.
        f30 = self.costFeatureExtractor3(f30)
        cost3 = torch.cat( (cost3, f30), 1 )

        # Disparity.
        # disp3, upDisp3 = self.disp3(cost3, upDisp4)
        disp3, upDisp3, upFeat3 = self.disp3(cost3)

        disp3   = F.leaky_relu(disp3,   0.3, inplace=False)
        upDisp3 = F.leaky_relu(upDisp3, 0.3, inplace=False)

        # ========== Scale 2. ==========
        scale = 2

        # Warp.
        # warp2 = self.warp( f21, upDisp3 * self.params.amp * 0.5**scale )
        upDisp3 = upDisp3 * ( 2 )
        warp2 = self.warp( f21, upDisp3 / self.params.amp )

        # Normalization
        f20   = self.fn2(f20)
        warp2 = self.fn2(warp2)

        # Correlation.
        cost2 = self.corr2dm( f20, warp2 )
        cost2 = self.corrActivation( cost2 )

        # Concatenate.
        f20 = self.costFeatureExtractor2(f20)
        cost2 = torch.cat( (cost2, f20), 1 )

        # Disparity.
        disp2, upDisp2, dispRes2 = self.disp2(cost2, upDisp3)

        # ========== Scale 1. ==========
        scale = 1

        # Warp.
        # warp1 = self.warp( f11, upDisp2 * self.params.amp * 0.5**scale )
        upDisp2 = upDisp2 * ( 2 )
        warp1 = self.warp( f11, upDisp2 / self.params.amp )

        # Normalization
        f10   = self.fn1(f10)
        warp1 = self.fn1(warp1)

        # Correlation.
        cost1 = self.corr2dm( f10, warp1 )
        cost1 = self.corrActivation( cost1 )

        # Concatenate.
        f10 = self.costFeatureExtractor1(f10)
        cost1 = torch.cat( (cost1, f10), 1 )

        # Disparity.
        disp1, upDisp1, dispRes1 = self.disp1(cost1, upDisp2)
        upDisp1 = upDisp1 * 2

        # ========== Disparity refinement. ==========
        r10 = self.re1(gray0)

        dispRe0 = self.refine( r10, upDisp1 )
        disp0 = upDisp1 * ( 1 + 0.1 * dispRe0 )

        if ( self.training ):
            return disp0, disp1, disp2, disp3#, disp4
        else:
            return disp0, disp1, disp2, disp3#, disp4

    def one_shot(self, gray0, gray1, initDisp):
        """
        gray0 and gray1 are the warped, stacked features/images.
        initDisp is the initial disparity map having the same HxW dimension.
        """

        # Get the image size of initDisp.
        H = initDisp.size()[2]
        W = initDisp.size()[3]

        halfDisp = F.interpolate( initDisp, (H//2, W//2), mode="bilinear", align_corners=False ) * 1.0*(W//2)/W

        # Feature extraction.
        f10 = self.fe1(gray0)
        f11 = self.fe1(gray1)

        # Normalization.
        f10 = self.fn1(f10)
        f11 = self.fn1(f11)

        # Correlation.
        cost1 = self.corr2dm( f10, f11 )
        cost1 = self.corrActivation( cost1 )

        f10 = self.costFeatureExtractor1(f10)
        cost1 = torch.cat( (cost1, f10), 1 )

        # Disparity.
        disp1, upDisp1, dispRes1 = self.disp1(cost1, halfDisp)
        upDisp1 = upDisp1 * 2

        # ========== Disparity refinement. ==========
        r10 = self.re1(gray0)

        dispRe0 = self.refine( r10, upDisp1 )
        disp0 = upDisp1 * ( 1 + 0.1 * dispRe0 )

        return disp0, ( dispRe0, disp1, dispRes1, upDisp1 )

if __name__ == "__main__":
    print("Test PWCNetStereo.py")

    params = PWCNetStereoParams()

    pwcns = PWCNetStereo(params)

    print("pwcns has %d model parameters. " % ( \
        sum( [ p.data.nelement() for p in pwcns.parameters() ] ) ))

    modelDict = pwcns.state_dict()
    for item in modelDict:
        print("Layer {}. ".format(item))
