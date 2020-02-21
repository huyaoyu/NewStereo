
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
# And 
# 
# The work by Sun, et. al.: PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume
# The reference non-official source code is at
# https://github.com/RanhaoKang/PWC-Net_pytorch
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

    sys.path.insert(0, "/home/yaoyu/Projects/NewStereo/Trial/RecurrentStereo/Model")
    import CommonModel as cm
else:
    from . import CommonModel as cm

import Corr2D

class ConvExtractor(nn.Module):
    def __init__(self, inCh, outCh, lastActivation=None):
        super(ConvExtractor, self).__init__()

        moduleList = [ \
            cm.Conv_Half( inCh, outCh, activation=nn.LeakyReLU(0.1) ), \
            # cm.Conv_W( outCh, outCh, activation=nn.LeakyReLU(0.1) ), \
            # cm.Conv_W( outCh, outCh, activation=nn.LeakyReLU(0.1) ) \
            ]
        
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
    def __init__(self, inCh, interCh, maxDisp):
        super(PredictDisparity, self).__init__()

        self.model = nn.Sequential( \
            cm.Conv_W(inCh, interCh, activation=nn.LeakyReLU(0.1)), \
            nn.BatchNorm2d(interCh, track_running_stats=False), \
            cm.Conv_W(interCh, 1), \
            nn.Tanh() )

        self.maxDisp = maxDisp
        
        # self.model = \
        #     cm.Conv_W(inCh, 1, activation=nn.LeakyReLU(0.1))

    def forward(self, x):
        return self.model(x) * self.maxDisp

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
    def __init__(self, inCh, interChList, maxDisp, flagUp=True):
        super(Cost2DisparityAndFeatureRes, self).__init__()

        # The regulator.
        self.regulator = CostRegulator(inCh, interChList)

        cumCh = np.cumsum( interChList )

        # The disparity predictor.
        self.disp = PredictDisparity( inCh + cumCh[-1], inCh, maxDisp )

        # The up-sample model.
        if ( flagUp ):
            self.upDisp = UpDisparity()
        else:
            self.upDisp = None

    def forward(self, x, lastDisp):
        x = self.regulator(x)

        disp = self.disp(x)
        disp = disp + lastDisp

        if ( self.upDisp is not None ):
            upDisp = self.upDisp(disp)

            return disp, upDisp
        else:
            return disp, x

class EDRegression(nn.Module):
    def __init__(self, inCh):
        super(EDRegression, self).__init__()

        self.encoder0_In  = inCh
        self.encoder0_Out = 32
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
        self.decoder0_Out = 32

        # Encoder-decoder.
        self.e0 = cm.Conv_Half( self.encoder0_In, self.encoder0_Out, k=3, activation=cm.SelectedReLU() )
        self.e1 = cm.Conv_Half( self.encoder1_In, self.encoder1_Out, k=3, activation=cm.SelectedReLU() )
        self.e2 = cm.Conv_Half( self.encoder2_In, self.encoder2_Out, k=3, activation=cm.SelectedReLU() )

        self.d3 = cm.Conv_W( self.decoder3_In, self.decoder3_Out, k=3, activation=cm.SelectedReLU() )
        self.d2 = cm.Deconv_DoubleSize( self.decoder2_In, self.decoder2_Out, k=4, p=1, activation=cm.SelectedReLU() )
        self.d1 = cm.Deconv_DoubleSize( self.decoder1_In, self.decoder1_Out, k=4, p=1, activation=cm.SelectedReLU() )
        self.d0 = cm.Deconv_DoubleSize( self.decoder0_In, self.decoder0_Out, k=4, p=1, activation=cm.SelectedReLU() )

        # self.finalUp = cm.Deconv_DoubleSize( self.decoder0_Out, 64, k=4, p=1, activation=cm.SelectedReLU() )

        # By-pass.
        self.bp  = cm.Conv_W( self.encoder0_In, self.decoder0_Out, k=3, activation=cm.SelectedReLU() )
        
        # Regression.
        self.rg0 = cm.Conv_W( self.decoder0_Out, 16, k=3, activation=cm.SelectedReLU() )
        self.rg1 = cm.Conv_W( 16, 1, k=3 )

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
        # fd0 = self.finalUp(fd0)

        # By-pass.
        bp = self.bp(x)

        # Regression.
        disp0 = self.rg0( fd0 + bp.mul(0.1) )
        disp1 = self.rg1( disp0 )

        return disp1

class WarpByDisparity(nn.Module):
    def __init__(self):
        super(WarpByDisparity, self).__init__()

    def forward(self, x, disp):
        """
        This is adopted from the code of PWCNet.
        """
        
        B, C, H, W = x.size()

        # Mesh grid. 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)

        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)

        grid = torch.cat((xx,yy),1).float()

        if ( x.is_cuda ):
            grid = grid.cuda()

        vgrid = grid.clone()

        # import ipdb; ipdb.set_trace()

        # Only the x-coodinate is changed. 
        # Disparity values are always non-negative.
        vgrid[:, 0, :, :] = vgrid[:, 0, :, :] - disp.squeeze(1) # Disparity only has 1 channel. vgrid[:, 0, :, :] will only have 3 dims.

        # Scale grid to [-1,1]. 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)

        output = nn.functional.grid_sample(x, vgrid)
        
        mask = torch.ones(x.size())
        if ( x.is_cuda ):
            mask = mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        
        mask[mask<0.9999] = 0
        mask[mask>0]      = 1
        
        return output * mask

class HeadFeatureExtractor(nn.Module):
    def __init__(self, inCh, outCh):
        super(HeadFeatureExtractor, self).__init__()

        self.firstConv = cm.Conv_W(inCh, outCh, k=3, activation=cm.SelectedReLU())

        # self.conv0 = cm.Conv_W(outCh, outCh, k=3, activation=cm.SelectedReLU())
        # self.conv1 = cm.Conv_Half(   outCh,   outCh, k=3, activation=cm.SelectedReLU() )
        # self.conv2 = cm.Conv_Half(   outCh, 2*outCh, k=3, activation=cm.SelectedReLU() )

        self.res   = cm.ResBlock( outCh, k=3, activation=cm.SelectedReLU(), lastActivation=cm.SelectedReLU() )

        # self.deconv2 = cm.Deconv_DoubleSize( 2*outCh,   outCh, k=4, p=1, activation=cm.SelectedReLU() )
        # self.deconv1 = cm.Deconv_DoubleSize( 2*outCh,   outCh, k=4, p=1, activation=cm.SelectedReLU() )

        # self.lastConv = cm.Conv_W( 2*outCh, outCh, k=3, activation=cm.SelectedReLU() )

    def forward(self, x):
        x = self.firstConv(x)

        # f0 = self.conv0(x)
        # f1 = self.conv1(f0)
        # f2 = self.conv2(f1)

        r  = self.res(x)

        # d2 = self.deconv2(r)

        # d2 = torch.cat((d2, f1), 1)
        # d1 = self.deconv1( d2 )

        # d1 = torch.cat((d1, f0), 1)

        # f = self.lastConv(d1)

        return r

class RecurrentFeatureExtractor(nn.Module):
    def __init__(self, inCh, outCh, interCh):
        super(RecurrentFeatureExtractor, self).__init__()

        self.firstNormalization = FeatureNormalization(inCh)

        self.conv0 = cm.Conv_W(inCh, outCh, k=3, activation=cm.SelectedReLU())
        # self.conv1 = cm.Conv_Half(   interCh,   interCh, k=3, activation=cm.SelectedReLU() )
        # self.conv2 = cm.Conv_Half( 2*interCh, 4*interCh, k=3, activation=cm.SelectedReLU() )

        self.res   = cm.ResBlock( outCh, k=3, activation=cm.SelectedReLU(), lastActivation=cm.SelectedReLU() )

        # self.deconv2 = cm.Deconv_DoubleSize( 4*interCh, 2*interCh, k=4, p=1, activation=cm.SelectedReLU() )
        # self.deconv1 = cm.Deconv_DoubleSize(   interCh,   interCh, k=4, p=1, activation=cm.SelectedReLU() )

        # self.lastConv = cm.Conv_W( 2*interCh, outCh, k=3, activation=cm.SelectedReLU() )
        self.outputNormalization = FeatureNormalization(outCh)

    def forward(self, x):
        x = self.firstNormalization(x)

        f0 = self.conv0(x)
        # f1 = self.conv1(f0)
        # f2 = self.conv2(f1)

        # r  = self.res(f2)
        r  = self.res(f0)

        # d2 = self.deconv2(r)
        # d2 = torch.cat((d2, f1), 1)

        # d1 = self.deconv1( d2 )
        # d1 = self.deconv1( r )

        # d1 = torch.cat((d1, f0), 1)

        # f = self.lastConv(d1)

        # f = self.outputNormalization(f)
        f = self.outputNormalization(r)

        return f

class RecurrentBlock(nn.Module):
    def __init__(self, inCh, corrCh, maxDisp, corrKernelSize):
        super(RecurrentBlock, self).__init__()

        self.fe = RecurrentFeatureExtractor(inCh=inCh, outCh=corrCh, interCh=inCh)

        self.maxDisp    = maxDisp
        self.padding    = maxDisp
        self.kernelSize = corrKernelSize
        self.strideK    = 1
        self.strideD    = 1

        # Correlation.
        self.corr2dm = Corr2D.Corr2DM( self.maxDisp, \
            padding=self.padding, \
            kernelSize=self.kernelSize, \
            strideK=self.strideK, \
            strideD=self.strideD )

        nd = self.maxDisp + 1 + self.maxDisp
        self.corrActivation = cm.Conv_W(nd, nd, k=1)

        # Warp.
        self.warp = WarpByDisparity()

        # Disparity residual regression.
        interChList = [ 128, 96, 64, 32 ]
        self.disp = Cost2DisparityAndFeatureRes(nd + corrCh, interChList, self.maxDisp, flagUp=False)

    def forward(self, f0, f1, upsampledScaledDisp):
        f0 = self.fe(f0)
        f1 = self.fe(f1)

        f1 = self.warp(f1, upsampledScaledDisp)

        cost = self.corr2dm( f0, f1 )
        cost = self.corrActivation(cost)

        cost = torch.cat((cost, f0), 1)

        disp, notUsed = self.disp(cost, upsampledScaledDisp)

        return disp, notUsed

class FeatureDownsampler(nn.Module):
    def __init__(self, inCh, outCh):
        super(FeatureDownsampler, self).__init__()

        self.model = ConvExtractor(inCh, outCh)

    def forward(self, x):
        return self.model(x)

class DisparityUpsampler(nn.Module):
    def __init__(self, interCh):
        super(DisparityUpsampler, self).__init__()

        self.model = nn.Sequential( \
            cm.Deconv_DoubleSize(1, interCh, k=4, p=1, activation=None), \
            cm.Conv_W(interCh, 1, k=3, activation=None))
    
    def forward(self, x):
        return self.model(x)

class RecurrentStereoParams(object):
    def __init__(self):
        super(RecurrentStereoParams, self).__init__()

        self.flagGray = False

        self.headCh = 65
        self.baseCh = 32

        self.amp = 1

        self.maxDisp = 16

        # Correlation.
        # self.corrPadding    = self.maxDisp
        self.corrKernelSize = 1
        # self.corrStrideK    = 1
        # self.corrStrideD    = 1

    def set_max_disparity(self, md):
        assert md > 0

        self.maxDisp = int( md )
        self.corrPadding = self.maxDisp

class RecurrentStereo(nn.Module):
    def __init__(self, params):
        super(RecurrentStereo, self).__init__()

        self.params = params

        if ( not self.params.flagGray ):
            raise Exception("Only support grayscale at the moment.")

        # Feature extractor.
        self.headFE = HeadFeatureExtractor( self.params.headCh, self.params.baseCh )
        self.downsampler0 = FeatureDownsampler(   self.params.baseCh,   self.params.baseCh )
        self.downsampler2 = FeatureDownsampler(   self.params.baseCh, 2*self.params.baseCh )
        self.downsampler3 = FeatureDownsampler( 2*self.params.baseCh, 2*self.params.baseCh )

        self.upsampler1 = DisparityUpsampler(4)
        self.upsampler2 = DisparityUpsampler(4)
        self.upsampler3 = DisparityUpsampler(4)

        # Recurrent block.
        self.rb0 = RecurrentBlock( self.params.baseCh, self.params.baseCh, \
            self.params.maxDisp, self.params.corrKernelSize )

        self.rb2 = RecurrentBlock( 2*self.params.baseCh, 2*self.params.baseCh, \
            self.params.maxDisp, self.params.corrKernelSize )

        # self.refine = DisparityRefineDilated( nd + 32 + chFeat )
        self.refine = EDRegression( self.params.baseCh + 1 )

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

    def forward(self, stack0, stack1, dispSlice=None):
        B, C, H, W = stack0.size()
        
        # Feature extraction.
        f00 = self.headFE( stack0 )
        f01 = self.headFE( stack1 )

        f10 = self.downsampler0( f00 )
        f11 = self.downsampler0( f01 )

        f20 = self.downsampler2( f10 )
        f21 = self.downsampler2( f11 )

        f30 = self.downsampler3( f20 )
        f31 = self.downsampler3( f21 )

        # f40 = self.downsampler( f30 )
        # f41 = self.downsampler( f31 )

        # import ipdb; ipdb.set_trace()

        # # ========== Scale 4. ========== 
        # scale = 4

        # H4 = f40.size()[2]
        # W4 = f40.size()[3]

        # # Temporary disparity.
        # dispT = torch.zeros((B, 1, H4, W4), dtype=torch.float32).cuda()
        # dispT.requires_grad = False

        # disp4, _ = self.rb( f40, f41, dispT )

        # ========== Scale 3. ==========
        scale = 3

        H3 = f30.size()[2]
        W3 = f30.size()[3]

        # Temporary disparity.
        dispT = torch.zeros((B, 1, H3, W3), dtype=torch.float32).cuda()
        dispT.requires_grad = False

        # disp4Up = F.interpolate( disp4, (H3, W3), mode="bilinear", align_corners=False ) * 2

        # disp3, _ = self.rb( f30, f31, disp4Up )
        disp3, _ = self.rb2( f30, f31, dispT )
        disp3 = F.relu(disp3, inplace=True)

        # ========== Scale 2. ==========
        scale = 2

        H2 = f20.size()[2]
        W2 = f20.size()[3]

        # disp3Up = F.interpolate( disp3, (H2, W2), mode="bilinear", align_corners=False ) * 2
        disp3Up = self.upsampler3( disp3 ) * 2

        disp2, _ = self.rb2( f20, f21, disp3Up )
        disp2 = F.relu(disp2, inplace=True)

        # ========== Scale 1. ==========
        scale = 1

        H1 = f10.size()[2]
        W1 = f10.size()[3]

        # disp2Up = F.interpolate( disp2, (H1, W1), mode="bilinear", align_corners=False ) * 2
        disp2Up = self.upsampler2( disp2 ) * 2

        disp1, _ = self.rb0( f10, f11, disp2Up )

        # ========== Scale 0. ==========
        scale = 0

        H0 = f00.size()[2]
        W0 = f00.size()[3]

        # disp1Up = F.interpolate( disp1, (H0, W0), mode="bilinear", align_corners=False ) * 2
        disp1Up = self.upsampler1( disp1 ) * 2

        disp0, _ = self.rb0( f00, f01, disp1Up )

        # ========== Disparity refinement. ==========

        dispRe0 = self.refine( torch.cat((f00, disp0), 1) )
        disp0 = disp0 + dispRe0

        if ( self.training ):
            return disp0, disp1, disp2, disp3#, disp4
        else:
            return disp0, disp1, disp2, disp3#, disp4

if __name__ == "__main__":
    print("Test RecurrentStereo.py")

    params = RecurrentStereoParams()
    params.flagGray = True

    rs = RecurrentStereo(params)

    print("RS has %d model parameters. " % ( \
        sum( [ p.data.nelement() for p in rs.parameters() ] ) ))

    modelDict = rs.state_dict()
    for item in modelDict:
        print("Layer {}. ".format(item))
