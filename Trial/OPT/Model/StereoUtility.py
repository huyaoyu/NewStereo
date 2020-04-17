
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

if ( __name__ == "__main__" ):
    import sys

    sys.path.insert(0, "/home/yaoyu/Projects/NewStereo/Trial/PRR/Model")
    import CommonModel as cm
else:
    from . import CommonModel as cm

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