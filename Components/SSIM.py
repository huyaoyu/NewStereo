from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Relative module.
from TorchBridge import cv_2_tensor

# CommonPython package.
from CommonPython.ImageWrite.ImageWrite import \
    write_float_image_normalized, write_float_image_normalized_clip, write_float_image_plt, write_float_image_plt_clip

def is_odd(x):
    """
    Return True if x is an odd number.
    """

    if ( x - int(x) != 0 ):
        return False
    
    if ( int(x) % 2 == 1 ):
        return True
    else:
        return False

def gaussian_kernel_1d( w, s ):
    """
    Creat a 1D Gaussian kernel with window size of w and sigma s.
    w must be an positive odd number with value at least 3.
    s must be a positive number.
    """

    if ( w < 3 ):
        raise Exception("w should be larger than or equal to 3. w = {}.".format(w))

    if ( not is_odd(w) ):
        raise Exception("w is not an odd number. w = {}.".format(w))

    if ( s <= 0 ):
        raise Exception("s should be positive. s = {}.".format(s))

    # Initial check done.

    # Force w to be an integer.
    w = int(w)

    # The relative coordinate.
    x = np.linspace( 1, w, w ).astype(np.float32) - (w // 2 + 1 )

    # Gaussian.
    e = np.exp( -x**2 / ( 2 * s**2 ) )
    g = torch.from_numpy(e)

    return g / g.sum()

def gaussian_kernel_2d(w, s):
    """
    Create a 2D Gaussian kernel.
    w must be an positive odd number with value at least 3.
    s must be a positive number.
    """

    # Create 1D Gaussian kernel.
    g = gaussian_kernel_1d(w, s)

    # Make g to be a column vector.
    g = g.unsqueeze(1)

    # Make a window
    gg = g.mm( g.t() )

    return gg

def uniform_kernel_2d(w):
    w = int(w)

    a = w * w

    return torch.ones((w, w), dtype=torch.float32) / a

class SSIM(torch.nn.Module):
    def __init__(self, w=11, sigma=1.5, channel=1, flagCuda=True):
        super( SSIM, self ).__init__()

        self.w       = w # Window size.
        self.padding = w // 2
        self.sigma   = sigma
        self.channel = channel # Will be used as groups.

        self.kernel = torch.autograd.Variable( \
            gaussian_kernel_2d( self.w, self.sigma ).expand(self.channel, 1, self.w, self.w).contiguous(), \
            requires_grad=False )

        # self.kernel = torch.autograd.Variable( \
        #     uniform_kernel_2d( self.w ).expand(self.channel, 1, self.w, self.w).contiguous(), \
        #     requires_grad=False )

        if ( flagCuda ):
            self.kernel = self.kernel.cuda()

        self.C0 = 0.01**2
        self.C1 = 0.03**2

    def forward(self, img_0, img_1):
        mu_0 = F.conv2d( img_0, self.kernel, padding=self.padding, groups=self.channel )
        mu_1 = F.conv2d( img_1, self.kernel, padding=self.padding, groups=self.channel )

        mu_0s = mu_0.pow(2.0)
        mu_1s = mu_1.pow(2.0)

        mu_01 = mu_0 * mu_1

        sigma_0s = F.conv2d( img_0 * img_0, self.kernel, padding=self.padding, groups=self.channel ) - mu_0s
        sigma_1s = F.conv2d( img_1 * img_1, self.kernel, padding=self.padding, groups=self.channel ) - mu_1s

        sigma_01 = F.conv2d( img_0 * img_1, self.kernel, padding=self.padding, groups=self.channel ) - mu_01

        F.relu( sigma_0s, inplace=True )
        F.relu( sigma_1s, inplace=True )
        F.relu( sigma_01, inplace=True )

        ssim = ( 2.0 * mu_01 + self.C0 ) * ( 2.0 * sigma_01 + self.C1 ) / ( ( mu_0s + mu_1s + self.C0 ) * ( sigma_0s + sigma_1s + self.C1 ) )

        if ( 1 != ssim.size()[1] ):
            # Average across the channel dimension.
            return ssim.mean(axis=1).unsqueeze(1)
        else:
            return ssim

if __name__ == "__main__":
    print("Test SSIM.py")

    print("A length 5 1D Gaussian kernel: g1D = ")
    g1D = gaussian_kernel_1d(5, 2)
    print(g1D)
    print("g1D.type() = {}.".format(g1D.type()))
    print("g1D.sum() = {}.".format(g1D.sum()))

    print("A 5x5 2D Gaussian kernel: ")
    g2D = gaussian_kernel_2d(5, 2)
    print(g2D)
    print("g2D.type() = {}.".format(g2D.type()))
    print("g2D.sum() = {}.".format(g2D.sum()))

    img_0 = torch.rand((1000, 1000), dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    img_1 = torch.rand((1000, 1000), dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

    ssim = SSIM()
    res = ssim(img_0, img_1)

    print("ssim(img_0, img_1) = {}.".format( res ))
    print("res.size() = {}.".format(res.size()))

    res = ssim(img_0, img_0)
    print("ssim(img_0, img_0) = {}.".format(res))

    # Test on real images.
    imgFn_0 = "/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_L_VC.png"
    imgFn_1 = "/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_R_VC.png"

    # Open the two images.
    img_0 = cv2.imread(imgFn_0, cv2.IMREAD_UNCHANGED)
    img_1 = cv2.imread(imgFn_1, cv2.IMREAD_UNCHANGED)

    # Convert the input image to gray image.
    imgGray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY).reshape( ( img_0.shape[0], img_0.shape[1] ,1) )
    imgGray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY).reshape( ( img_1.shape[0], img_1.shape[1] ,1) )

    print("Read \n%s and\n%s" % ( imgFn_0, imgFn_1 ))

    # Convert the image into torch Tensor.
    img_0 = cv_2_tensor(img_0, dtype=np.float32, flagCuda=True)
    img_1 = cv_2_tensor(img_1, dtype=np.float32, flagCuda=True)

    # SSIM.
    ssim3 = SSIM(channel=3)
    res = ssim3(img_0, img_1)

    print("res.size() = {}.".format(res.size()))
    res = res.squeeze(0).squeeze(0).cpu().numpy()

    # Statistics.
    print("res: mean = {}, max = {}, min = {}.".format( res.mean(), res.max(), res.min() ))

    # Save res as an image.
    write_float_image_normalized_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_LR_VC_SSIM3.png", res, -1.0, 1.0)

    # Plot by matplotlib.
    write_float_image_plt_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_LR_VC_SSIM3_PLT.png", res, -1.0, 1.0)

    # Convert the gray image into torch Tensor.
    imgGray_0 = cv_2_tensor(imgGray_0, dtype=np.float32, flagCuda=True)
    imgGray_1 = cv_2_tensor(imgGray_1, dtype=np.float32, flagCuda=True)

    # SSIM.
    res = ssim(imgGray_0, imgGray_1)
    res = res.squeeze(0).squeeze(0).cpu().numpy()

    # Statistics.
    print("res: mean = {}, max = {}, min = {}.".format( res.mean(), res.max(), res.min() ))

    # Save res as an image.
    write_float_image_normalized_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_LR_VC_SSIM1.png", res, -1.0, 1.0)

    # Plot by matplotlib.
    write_float_image_plt_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_LR_VC_SSIM1_PLT.png", res, -1.0, 1.0)

    # ======== Single pixel shift. ========

    print("Single pixel shift.")
    res = ssim3(img_0[:,:,:,:-1], img_0[:,:,:,1:])

    print("res.size() = {}.".format(res.size()))
    res = res.squeeze(0).squeeze(0).cpu().numpy()

    # Statistics.
    print("res: mean = {}, max = {}, min = {}.".format( res.mean(), res.max(), res.min() ))

    # Save res as an image.
    write_float_image_normalized_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_LR_VC_SSIM3_Single.png", res, -1.0, 1.0)

    # Plot by matplotlib.
    write_float_image_plt_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_LR_VC_SSIM3_Single_PLT.png", res, -1.0, 1.0)

    # ======== 2-pixel shift. ========

    print("2-pixel shift.")
    res = ssim3(img_0[:,:,:,:-2], img_0[:,:,:,2:])

    print("res.size() = {}.".format(res.size()))
    res = res.squeeze(0).squeeze(0).cpu().numpy()

    # Statistics.
    print("res: mean = {}, max = {}, min = {}.".format( res.mean(), res.max(), res.min() ))

    # Save res as an image.
    write_float_image_normalized_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_LR_VC_SSIM3_S2.png", res, -1.0, 1.0)

    # Plot by matplotlib.
    write_float_image_plt_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_LR_VC_SSIM3_S2_PLT.png", res, -1.0, 1.0)

    # ======== Blank 1024 pixels. ========

    print("1024-pixel blank.")

    # Create a new tensor.
    img_0_1024 = img_0.squeeze(0).cpu().numpy()
    img_0_1024[:,:,:1024] = 0
    img_0_1024 = torch.from_numpy(img_0_1024).unsqueeze(0).cuda()

    res = ssim3(img_0, img_0_1024)

    print("res.size() = {}.".format(res.size()))
    res = res.squeeze(0).squeeze(0).cpu().numpy()
    resROI = res[:, 1024:]

    # Statistics.
    print("res: mean = {}, max = {}, min = {}.".format( resROI.mean(), resROI.max(), resROI.min() ))

    # Save res as an image.
    write_float_image_normalized_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_LR_VC_SSIM3_B1024.png", res, -1.0, 1.0)

    # Plot by matplotlib.
    write_float_image_plt_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/000122/000122_LR_VC_SSIM3_B1024_PLT.png", res, -1.0, 1.0)

    # ======== Test the image warping. ========

    # Test on real images.
    imgFn_0 = "/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/200122/Rectified_L_color.jpg"
    imgFn_1 = "/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/200122/Rectified_R_color_warped.png"

    # Open the two images.
    img_0 = cv2.imread(imgFn_0, cv2.IMREAD_UNCHANGED)
    img_1 = cv2.imread(imgFn_1, cv2.IMREAD_UNCHANGED)

    # Convert the input image to gray image.
    imgGray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    imgGray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

    # Convert the image into torch Tensor.
    img_0 = cv_2_tensor(img_0, dtype=np.float32, flagCuda=True)
    img_1 = cv_2_tensor(img_1, dtype=np.float32, flagCuda=True)

    print("Read \n%s and\n%s" % ( imgFn_0, imgFn_1 ))
    res = ssim3(img_0, img_1).squeeze(0).squeeze(0).cpu().numpy()
    resROI = res[:,1024:]

    # Statistics.
    print("resROI: mean = {}, max = {}, min = {}.".format( resROI.mean(), resROI.max(), resROI.min() ))

    # Plot by matplotlib.
    write_float_image_plt_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/200122/200122_LR_SSIM3_PLT.png", res, -1.0, 1.0)

    # Convert the gray image into torch Tensor.
    imgGray_0 = cv_2_tensor(imgGray_0, dtype=np.float32, flagCuda=True)
    imgGray_1 = cv_2_tensor(imgGray_1, dtype=np.float32, flagCuda=True)

    res = ssim(imgGray_0, imgGray_1).squeeze(0).squeeze(0).cpu().numpy()
    resROI = res[:, 1024:]

    # Statistics.
    print("resROI: mean = {}, max = {}, min = {}.".format( resROI.mean(), resROI.max(), resROI.min() ))

    # Plot by matplotlib.
    write_float_image_plt_clip("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/200122/200122_LR_SSIM1_PLT.png", res, -1.0, 1.0)