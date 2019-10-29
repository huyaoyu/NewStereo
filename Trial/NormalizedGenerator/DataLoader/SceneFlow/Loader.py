
from __future__ import print_function

# Most of the content of this file is copied from PSMNet.
# https://github.com/JiaRenChang/PSMNet

import copy
import cv2
import numpy as np
from PIL import Image, ImageOps
import random

import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms

if ( __name__ == "__main__" ):
    import sys

    sys.path.insert(0, "/home/yaoyu/Projects/NewStereo/Trial/NormalizedGenerator/DataLoader")
    import IO
    import PreProcess
    import TorchTensorUtils
else:
    from . import IO
    from .. import PreProcess
    from .. import TorchTensorUtils

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def cv2_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def disparity_loader(path):
    return IO.readPFM(path)

def generate_2D_Gaussian(n):
    """
    Create a 2D nxn NumPy array which prepresents a 0-1 Gaussian distribution.
    n: Positive integer, the patch size.

    This function generate a distribution without the 1.0/sqrt(2.0 pi) coefficient.
    """

    if ( n <= 0 ):
        raise Exception("n must be a positive integer. n = {}. ".format(n))

    x = np.linspace(0, n, num=n, endpoint=False, dtype=np.float32)
    y = np.linspace(0, n, num=n, endpoint=False, dtype=np.float32)

    x, y = np.meshgrid( x, y )

    half = (n-1)/2

    x = (x - half) / half
    y = (y - half) / half

    rs = x**2 + y**2

    return np.exp( -rs/2.0 ).astype(np.float32)

class myImageFolder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, \
        loader=cv2_loader, dploader= disparity_loader, \
        preprocessorImg=None, preprocessorDisp=None, \
        cropSize=(0,0), gNoiseWidth=-1):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

        self.preprocessorImg  = preprocessorImg
        self.preprocessorDisp = preprocessorDisp
        self.cropSize = cropSize # (h, w)

        self.dispWhiteNoiseLevel = 0.05

        # Gaussian noise for the disparity.
        if ( gNoiseWidth < 0 ):
            # Disabled.
            self.dispGaussianNoiseWidth = -1
        elif ( gNoiseWidth == 0 ):
            if ( 0 == self.cropSize[0] ):
                raise Exception("Gaussian noise for the disparity could not applied with the default width on cropSize smaller than the 8x Gaussian window width. self.cropSize = {}, gNoiseWidth = {}. ".format( self.cropSize, gNoiseWidth ))
            
            # 1/4 the cropSize[0];
            self.dispGaussianNoiseWidth = int(self.cropSize[0] / 8)
        else:
            if ( gNoiseWidth > self.cropSize[0] or gNoiseWidth > self.cropSize[1] ):
                raise Exception("Gaussian noise for the disparity could not applied on cropSize smaller than the Gaussian window width. self.cropSize = {}, gNoiseWidth = {}. ".format( self.cropSize, gNoiseWidth ))

            self.dispGaussianNoiseWidth = gNoiseWidth

        if ( self.dispGaussianNoiseWidth > 0 ):
            self.dispGaussianWindow = generate_2D_Gaussian( self.dispGaussianNoiseWidth )
        else:
            self.dispGaussianWindow = None

    def self_normalize(self, x):
        """
        x is an OpenCV mat.
        """

        oriMin = x.min()
        x = x - oriMin
        step = x.max()
        
        return x / step, oriMin, step

    def ref_normalize(self, x, m, s):
        """
        x is an OpenCV mat.
        m and s are the 'oriMin' and 'step' outputs of self_normalize().
        """

        x = x - m
        x = x / s
        x = np.clip(x, 0.0, 1.0)

        return x

    def random_crop_image_and_disparity(self, img0, img1, disp):
        """
        imgp, img1, and disp are assumed to be NumPy arrays with the same shape.
        And they are assumbed to be OpenCV mat objects, meaning the order of dimensions is
        height, width, and channel.
        """

        # Allowed indices.
        ah = img0.shape[0] - self.cropSize[0]
        aw = img1.shape[1] - self.cropSize[1]

        if ( ah < 0 ):
            raise Exception("img0.shape[0] < self.cropSize[0]. img0.shape[0] = {}, self.cropSize[0] = {}. ".format( img0.shape[0], self.cropSize[0] ))

        if ( aw < 0 ):
            raise Exception("img0.shape[1] < self.cropSize[1]. img0.shape[1] = {}, self.cropSize[1] = {}. ".format( img0.shape[1], self.cropSize[1] ))

        # Get two random numbers.
        ah = ah + 1
        aw = aw + 1

        ch = np.random.randint(0, ah)
        cw = np.random.randint(0, aw)

        return img0[ch:ch+self.cropSize[0], cw:cw+self.cropSize[1]], \
               img1[ch:ch+self.cropSize[0], cw:cw+self.cropSize[1]], \
               disp[ch:ch+self.cropSize[0], cw:cw+self.cropSize[1]]

    def center_crop_image_and_disparity(self, img0, img1, disp):
        """
        imgp, img1, and disp are assumed to be NumPy arrays with the same shape.
        And they are assumbed to be OpenCV mat objects, meaning the order of dimensions is
        height, width, and channel.
        """

        # Allowed indices.
        ah = img0.shape[0] - self.cropSize[0]
        aw = img1.shape[1] - self.cropSize[1]

        if ( ah < 0 ):
            raise Exception("img0.shape[0] < self.cropSize[0]. img0.shape[0] = {}, self.cropSize[0] = {}. ".format( img0.shape[0], self.cropSize[0] ))

        if ( aw < 0 ):
            raise Exception("img0.shape[1] < self.cropSize[1]. img0.shape[1] = {}, self.cropSize[1] = {}. ".format( img0.shape[1], self.cropSize[1] ))

        # Get the center crop.
        ah = ah + 1
        aw = aw + 1

        ch = int( ah / 2 )
        cw = int( aw / 2 )

        return img0[ch:ch+self.cropSize[0], cw:cw+self.cropSize[1]], \
               img1[ch:ch+self.cropSize[0], cw:cw+self.cropSize[1]], \
               disp[ch:ch+self.cropSize[0], cw:cw+self.cropSize[1]]
    
    def half_size(self, x):
        """
        Assuming x is an OpenCV mat object.
        """

        h = int(x.shape[0]/2)
        w = int(x.shape[0]/2)

        return cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)

    def add_disparity_noise(self, disp):
        n = ( np.random.rand(disp.shape[0], disp.shape[1]) * 2 - 1 ) * self.dispWhiteNoiseLevel
        n = n.astype( np.float32 )

        disp = disp *(1 + n)
        
        return np.clip(disp, 1.0, disp.max())

    def add_disparity_random_Gaussian_noise(self, disp):
        """
        disp is OpenCV mat.
        """
        
        if ( self.dispGaussianWindow is None ):
            return disp

        # Get a random starting index.
        n   = disp.shape[0] - self.dispGaussianNoiseWidth
        idx = np.random.randint(0, n)

        # Get a random scale.
        s = ( np.random.rand() * 2 - 1 ) * 0.5

        # Deep copy.
        x = copy.deepcopy(disp)

        # The in-place indexing assignment.
        x[ idx:idx+self.dispGaussianNoiseWidth, idx:idx+self.dispGaussianNoiseWidth ] = \
            x[ idx:idx+self.dispGaussianNoiseWidth, idx:idx+self.dispGaussianNoiseWidth ] * ( 1 + s * self.dispGaussianWindow.astype(np.float32) )

        return x
    
    def disparity_2_tensor(self, x):
        """
        x is a single channel OpenCV mat object.
        """

        return torch.from_numpy(x.astype(np.float32))

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        imgL = self.loader(left)
        # import ipdb; ipdb.set_trace()
        imgR = self.loader(right)
        dispL, scaleL = self.dploader(disp_L)
        dispL = np.ascontiguousarray(dispL, dtype=np.float32)

        if self.training:  
            # cv2 compatible crop.
            imgL, imgR, dispL = \
                self.random_crop_image_and_disparity( imgL, imgR, dispL )

            # Disparity.
            dispLH = self.half_size(dispL)
            dispLH = self.add_disparity_random_Gaussian_noise(dispLH)
            dispLH = self.add_disparity_noise(dispLH)
            dispLH, nm, ns = self.self_normalize(dispLH)
            dispL = self.ref_normalize(dispL, nm, ns)

            # Downsampled version of imgL.
            imgLH = cv2.pyrDown(imgL, dstsize=(dispLH.shape[1], dispLH.shape[0]))
        else:
            # cv2 compatible crop.
            imgL, imgR, dispL = \
                self.center_crop_image_and_disparity( imgL, imgR, dispL )

            # Disparity.
            dispLH = self.half_size(dispL)
            dispLH = self.add_disparity_random_Gaussian_noise(dispLH)
            dispLH = self.add_disparity_noise(dispLH)
            dispLH, nm, ns = self.self_normalize(dispLH)
            dispL = self.ref_normalize(dispL, nm, ns)

            # Downsampled version of imgL.
            imgLH = cv2.pyrDown(imgL, dstsize=(dispLH.shape[1], dispLH.shape[0]))

        # Image pre-processing.
        if ( self.preprocessorImg is not None ):
            imgL  = self.preprocessorImg(imgL)
            imgR  = self.preprocessorImg(imgR)
            imgLH = self.preprocessorImg(imgLH)
        
        # Disparity pre-processing.
        if ( self.preprocessorDisp is not None ):
            dispL  = self.preprocessorDisp(dispL)
            dispLH = self.preprocessorDisp(dispLH)

        return imgL, imgR, dispL, dispLH, imgLH

    def __len__(self):
        return len(self.left)

class inferImageFolder(data.Dataset):
    def __init__(self, left, right, Q, \
        loader=cv2_loader, preprocessor=None, \
        cropSize=(0,0)):
 
        self.left   = left
        self.right  = right
        self.Q      = Q
        self.loader = loader

        # Modified.
        self.preprocessor  = preprocessor
        self.cropSize      = cropSize

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]

        left_img = self.loader(left)
        right_img = self.loader(right)

        # w, h = left_img.size

        w = left_img.shape[1]
        h = left_img.shape[0]

        if ( self.cropSize[0] <= 0 or self.cropSize[1] <= 0 ):
            ch, cw = h, w
        else:
            ch, cw = self.cropSize[0], self.cropSize[1]

        # left_img  = left_img.crop( (w-cw, h-ch, w, h))
        # right_img = right_img.crop((w-cw, h-ch, w, h))

        # cv2 compatible crop.
        left_img  = left_img[ h-ch:h, w-cw:w ]
        right_img = right_img[ h-ch:h, w-cw:w ]

        if ( self.preprocessor is not None ):
            left_img  = self.preprocessor(left_img)
            right_img = self.preprocessor(right_img)

        # Load the Q matrix.
        Q = self.Q[index]

        return left_img, right_img, torch.from_numpy( np.loadtxt( Q, dtype=np.float32 ) )

    def __len__(self):
        return len(self.left)

if __name__ == "__main__":
    import glob
    import os

    leftFiles  = sorted( glob.glob("/home/yaoyu/expansion/StereoData/SceneFlowSample/FlyingThings3D/RGB_cleanpass/left/*.png") )
    rightFiles = sorted( glob.glob("/home/yaoyu/expansion/StereoData/SceneFlowSample/FlyingThings3D/RGB_cleanpass/right/*.png") )
    dispFiles  = sorted( glob.glob("/home/yaoyu/expansion/StereoData/SceneFlowSample/FlyingThings3D/disparity/*.pfm") )

    outDir = "/home/yaoyu/Transient/NormalizedGeneratorDataLoader"

    # Test the output directory.
    if ( not os.path.isdir(outDir) ):
        os.makedirs( outDir )

    dl = myImageFolder(leftFiles, rightFiles, dispFiles, training=True, cropSize=(512, 512), gNoiseWidth=0)

    imgL, imgR, dispL, dispLH = dl[0]

    # Convert imgL and imgR to dummy tensors.
    imgL = torch.from_numpy( imgL.astype(np.float32) ).permute((2,0,1))
    imgR = torch.from_numpy( imgR.astype(np.float32) ).permute((2,0,1))

    # Convert dispL and dispLH to dummy tensors.
    dispL  = torch.from_numpy( dispL.astype(np.float32) )
    dispLH = torch.from_numpy( dispLH.astype(np.float32) )
    
    # Save the data to the file system.
    TorchTensorUtils.save_tensor_image( "%s/ImgL.png" % (outDir), imgL )
    TorchTensorUtils.save_tensor_image( "%s/ImgR.png" % (outDir), imgR )
    TorchTensorUtils.save_tensor_image_single_channel_normalized( "%s/dispL.png" % (outDir), dispL )
    TorchTensorUtils.save_tensor_image_single_channel_normalized( "%s/dispLH.png" % (outDir), dispLH )
    