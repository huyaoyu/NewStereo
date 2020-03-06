
from __future__ import print_function

# Most of the content of this file is copied from PSMNet.
# https://github.com/JiaRenChang/PSMNet

import copy
import cv2
import numpy as np
import os
from PIL import Image, ImageOps
import time

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

if ( __name__ == "__main__" ):
    import sys

    sys.path.insert(0, "/home/yaoyu/Projects/NewStereo/Trial/SmallSizeStereo/DataLoader")
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

def test_file(file):
    if ( not os.path.isfile(file) ):
        raise Exception("%s does not exist. " % (file))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    test_file(path)
    return Image.open(path).convert('RGB')

def cv2_loader(path):
    test_file(path)
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def cv2_loader_float32(path):
    test_file(path)
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

def convert_2_gray_gradx(img):
    # Grayscale.
    if ( 3 == len(img.shape) ):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gradient.
    grad = cv2.Sobel(img, cv2.CV_32FC1, 1, 0)

    return img, grad

def cv2_loader_gray_gradx_float(path):
    # The image.
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Grayscale and gradient.
    gray, grad = convert_2_gray_gradx(img)

    return gray.astype(cv2.CV_32FC1), grad

def disparity_loader(path):
    test_file(path)
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
        preprocessorImg=None, preprocessorGrad=None, preprocessorDisp=None, \
        cropSize=(0,0), newSize=(0,0), gNoiseWidth=-1):
 
        self.left     = left
        self.right    = right
        self.disp_L   = left_disparity
        self.loader   = loader
        self.dploader = dploader
        self.training = training

        self.preprocessorImg  = preprocessorImg
        self.preprocessorGrad = preprocessorGrad
        self.preprocessorDisp = preprocessorDisp
        self.cropSize = cropSize # (h, w)
        self.newSize  = newSize # (h, w)

        self.dispWhiteNoiseLevel = 0.05

        self.flagGray  = False
        self.flagGradX = False

    def enable_gray(self):
        self.flagGray = True

    def enable_grad_x(self):
        self.flagGray  = True
        self.flagGradX = True

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

        ch = torch.randint(0, ah, (1, )).item()
        cw = torch.randint(0, aw, (1, )).item()

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
    
    def resize_images_and_disparity(self, img0, img1, disp, newSize):
        """
        newSize is in order of (H, W).
        """

        img0 = cv2.resize(img0, (newSize[1], newSize[0]), interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, (newSize[1], newSize[0]), interpolation=cv2.INTER_LINEAR)

        wOri = disp.shape[1]

        disp = cv2.resize(disp, (newSize[1], newSize[0]), interpolation=cv2.INTER_LINEAR) * (1.0*newSize[1])/wOri

        return img0, img1, disp

    def half_size(self, x):
        """
        Assuming x is an OpenCV mat object.
        """

        h = int(x.shape[0]/2)
        w = int(x.shape[0]/2)

        return cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)

    def add_disparity_noise(self, disp):
        # randomBase = np.random.rand(disp.shape[0], disp.shape[1])
        randomBase = torch.rand( ( disp.shape[0], disp.shape[1] ), dtype=torch.float32 ).numpy()
        # print("randomBase[0,0] = %f, pid = %d. " % (randomBase[0, 0], os.getpid()))

        n = ( randomBase * 2 - 1 ) * self.dispWhiteNoiseLevel
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
        n0   = disp.shape[0] - self.dispGaussianNoiseWidth
        n1   = disp.shape[1] - self.dispGaussianNoiseWidth
        idx0 = torch.randint( 0, n0, (1, ) ).item()
        idx1 = torch.randint( 0, n1, (1, ) ).item()

        # Get a random scale.
        s = ( torch.rand(1).item() * 2 - 1 ) * 0.5

        # Deep copy.
        x = copy.deepcopy(disp)

        # The in-place indexing assignment.
        x[ idx0:idx0+self.dispGaussianNoiseWidth, idx1:idx1+self.dispGaussianNoiseWidth ] = \
            x[ idx0:idx0+self.dispGaussianNoiseWidth, idx1:idx1+self.dispGaussianNoiseWidth ] * ( 1 + s * self.dispGaussianWindow.astype(np.float32) )

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

        if ( self.cropSize[0] != 0 and self.cropSize[1] != 0 ):
            if self.training:  
                # cv2 compatible crop.
                imgL, imgR, dispL = \
                    self.random_crop_image_and_disparity( imgL, imgR, dispL )
            else:
                # cv2 compatible crop.
                imgL, imgR, dispL = \
                    self.center_crop_image_and_disparity( imgL, imgR, dispL )

        # Resize.
        if ( self.newSize[0] != 0 and self.newSize[1] != 0):
            imgL, imgR, dispL = \
                self.resize_images_and_disparity( imgL, imgR, dispL, self.newSize )

        # Grayscale and gradient.
        if ( self.flagGray ):
            imgL, gradL = convert_2_gray_gradx(imgL)
            imgR, gradR = convert_2_gray_gradx(imgR)

        # Image pre-processing.
        if ( self.preprocessorImg is not None ):
            imgL  = self.preprocessorImg(imgL.astype(np.float32))
            imgR  = self.preprocessorImg(imgR.astype(np.float32))
        
        if ( self.flagGradX and self.preprocessorGrad is not None ):
            gradL = self.preprocessorGrad(gradL)
            gradR = self.preprocessorGrad(gradR)
        
        # Disparity pre-processing.
        if ( self.preprocessorDisp is not None ):
            dispL  = self.preprocessorDisp(dispL)

        if ( self.flagGradX ):
            return {"img0": imgL, "img1": imgR, "disp0": dispL, "grad0": gradL, "grad1": gradR}
        else:
            return {"img0": imgL, "img1": imgR, "disp0": dispL, \
                "grad0": torch.zeros((1), dtype=torch.float), "grad1": torch.zeros((1), dtype=torch.float)}

    def __len__(self):
        return len(self.left)

# inferImageFolder is not ready yet.
class inferImageFolder(data.Dataset):
    def __init__(self, left, right, Q, \
        loader=cv2_loader, preprocessor=None, \
        cropSize=(0,0), newSize=(0,0)):
 
        self.left   = left
        self.right  = right
        self.Q      = Q
        self.loader = loader

        # Modified.
        self.preprocessor  = preprocessor
        self.cropSize      = cropSize
        self.newSize       = newSize

    def center_crop_images(self, img0, img1, cropSize):
        """
        imgp and img1 are assumed to be NumPy arrays with the same shape.
        And they are assumbed to be OpenCV mat objects, meaning the order of dimensions is
        height, width, and channel.
        """

        # Allowed indices.
        ah = img0.shape[0] - cropSize[0]
        aw = img1.shape[1] - cropSize[1]

        if ( ah < 0 ):
            raise Exception("img0.shape[0] < self.cropSize[0]. img0.shape[0] = {}, self.cropSize[0] = {}. ".format( img0.shape[0], cropSize[0] ))

        if ( aw < 0 ):
            raise Exception("img0.shape[1] < self.cropSize[1]. img0.shape[1] = {}, self.cropSize[1] = {}. ".format( img0.shape[1], cropSize[1] ))

        # Get the center crop.
        ah = ah + 1
        aw = aw + 1

        ch = int( ah / 2 )
        cw = int( aw / 2 )

        return img0[ch:ch+cropSize[0], cw:cw+cropSize[1]], \
               img1[ch:ch+cropSize[0], cw:cw+cropSize[1]]

    def resize_images(self, img0, img1, newSize):
        """
        newSize is in order of (H, W).
        """

        img0 = cv2.resize(img0, (newSize[1], newSize[0]), interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, (newSize[1], newSize[0]), interpolation=cv2.INTER_LINEAR)

        return img0, img1

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]

        img0 = self.loader(left)
        img1 = self.loader(right)

        if ( self.cropSize[0] > 0 and self.cropSize[1] > 0 ):
            img0, img1 = self.center_crop_images(img0, img1, self.cropSize)

        if ( self.newSize[0] > 0 and self.newSize[1] > 0 ):
            img0, img1 = self.resize_images(img0, img1, self.newSize)

        if ( self.preprocessor is not None ):
            img0 = self.preprocessor(img0)
            img1 = self.preprocessor(img1)

        # Load the Q matrix.
        Q = self.Q[index]

        return { "img0": img0, "img1": img1, "Q": torch.from_numpy( np.loadtxt( Q, dtype=np.float32 ) ) }

    def __len__(self):
        return len(self.left)

if __name__ == "__main__":
    # import glob
    # import os

    # leftFiles  = sorted( glob.glob("/home/yaoyu/temp/SceneFlowSample/FlyingThings3D/RGB_cleanpass/left/*.png") )
    # rightFiles = sorted( glob.glob("/home/yaoyu/temp/SceneFlowSample/FlyingThings3D/RGB_cleanpass/right/*.png") )
    # dispFiles  = sorted( glob.glob("/home/yaoyu/temp/SceneFlowSample/FlyingThings3D/disparity/*.pfm") )

    # outDir = "/home/yaoyu/temp/NewStereoData/SSS/DataLoader"

    # # Test the output directory.
    # if ( not os.path.isdir(outDir) ):
    #     os.makedirs( outDir )

    # dl = myImageFolder(leftFiles, rightFiles, dispFiles, training=True, cropSize=(512, 512), gNoiseWidth=0)

    # imgL, imgR, dispL, gradL, gradR = dl[0]

    # h = imgL.shape[0]
    # w = imgL.shape[1]

    # # Convert imgL and imgR to dummy tensors.
    # imgL = torch.from_numpy( imgL.astype(np.float32).reshape((h,w,1)) ).permute((2,0,1))
    # imgR = torch.from_numpy( imgR.astype(np.float32).reshape((h,w,1)) ).permute((2,0,1))

    # # Convert dispL to dummy tensors.
    # dispL  = torch.from_numpy( dispL.astype(np.float32) )

    # # Convert gradL and gradR to dummy tensors.
    # gradL = torch.from_numpy( gradL.reshape((h,w,1)) ).permute((2, 0, 1))
    # gradR = torch.from_numpy( gradR.reshape((h,w,1)) ).permute((2, 0, 1))
    
    # # Save the data to the file system.
    # TorchTensorUtils.save_tensor_image( "%s/ImgL.png" % (outDir), imgL )
    # TorchTensorUtils.save_tensor_image( "%s/ImgR.png" % (outDir), imgR )
    # TorchTensorUtils.save_tensor_image_single_channel_normalized( "%s/dispL.png" % (outDir), dispL )
    # TorchTensorUtils.save_tensor_image_single_channel_normalized( "%s/gradL.png" % (outDir), gradL )
    # TorchTensorUtils.save_tensor_image_single_channel_normalized( "%s/gradR.png" % (outDir), gradR )

    raise Exception("Not implemented.")
