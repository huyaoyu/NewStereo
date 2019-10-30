from __future__ import print_function

import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def is_odd(x):
    r = x%2

    if ( r != 1 ):
        return False
    else:
        return True

class Conv_W(nn.Module):
    def __init__(self, inCh, outCh, k=3):
        """
        k must be an odd number.
        """
        super(Conv_W, self).__init__()

        if ( not is_odd(k) ):
            raise Exception("k must be an odd number. k = {}. ".format(k))

        self.model = nn.Conv2d(inCh, outCh, kernel_size=k, stride=1, padding=k//2, dilation=1, bias=True)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, inCh, outCh, k, scale=None):
        super(ResBlock, self).__init__()

        self.model = nn.Sequential( \
            Conv_W(inCh, inCh, k), \
            nn.ReLU(True), \
            Conv_W(inCh, outCh, k) )
        
        self.scale = scale

    def forward(self, x):
        if ( self.scale is not None ):
            res = self.model(x).mul( self.scale )
        else:
            res = self.model(x)
        
        res += x

        return F.relu(res, inplace=True)

class ResPack(nn.Module):
    def __init__(self, inCh, outCh, n, k, scale=None):
        super(ResPack, self).__init__()

        m = []

        for i in range(n):
            m.append( ResBlock( inCh, inCh, k, scale ) )

        m.append( Conv_W( inCh, outCh, k ) )

        self.model = nn.Sequential(*m)

    def forward(self, x):
        res = self.model(x)
        res += x

        return F.relu(res, inplace=True)

class ReceptiveBranch(nn.Module):
    def __init__(self, inCh, outCh, r):
        super(ReceptiveBranch, self).__init__()

        self.model = nn.Sequential( \
            nn.AvgPool2d( kernel_size=r, stride=r, padding=0, ceil_mode=False, count_include_pad=True ), \
            nn.Conv2d( inCh, outCh, kernel_size=1, stride=1, padding=0, dilation=1, bias=True ), \
            nn.ReLU(True) )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    print("Test CommonModel.py")

    inCh  = 32
    outCh = 64
    k     = 3

    conv_w    = Conv_W(inCh, outCh, k)
    resBlock  = ResBlock( inCh, outCh, k )
    resPack   = ResPack( inCh, outCh, 2, k )
    recBranch = ReceptiveBranch( inCh, outCh, 4 )
