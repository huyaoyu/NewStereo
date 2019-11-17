import torch

import DispCorr

class DispCorrFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        s = DispCorr.forward(x)
        ctx.save_for_backward( x, s[0] )
        return s[0]

    @staticmethod
    def backward(ctx, grad):
        x, s = ctx.saved_variables

        output = DispCorr.backward( grad, s, x )

        return output[0]

class DispCorrM(torch.nn.Module):
    def __init__(self):
        super(DispCorrM, self).__init__()
    
    def forward(self, x):
        return DispCorrFunction.apply( x )
