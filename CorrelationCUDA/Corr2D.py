import torch

import Corr2D_ext

class Corr2DF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        s = Corr2D_ext.forward(x)
        ctx.save_for_backward( s[0] )
        return s[0]

    @staticmethod
    def backward(ctx, grad):
        sv = ctx.saved_variables

        output = Corr2D_ext.backward( grad, sv[0] )

        return output[0]

class Corr2DM(torch.nn.Module):
    def __init__(self):
        super(Corr2DM, self).__init__()
    
    def forward(self, x):
        return Corr2DF.apply( x )
