
from __future__ import print_function

# Most of the content of this file is copied from PSMNet.
# https://github.com/JiaRenChang/PSMNet

import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

#imagenet_stats = {'mean': [0.5, 0.5, 0.5],
#                   'std': [0.5, 0.5, 0.5]}

imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs

class SingleChannel(object):
    def __call__(self, x):
        return x[0].view( ( 1, x.size()[1], x.size()[2] ) )

class NormalizeRGB_OCV(object):
    def __init__(self, s):
        super(NormalizeRGB_OCV, self).__init__()
        
        self.s = s

    def __call__(self, x):
        """This is the OpenCV version. The order of the color channle is BGR. The order of dimension is HWC."""

        x = np.copy(x) * self.s

        # It is assumed that the data type of x is already floating point number.

        x[:, :, 0] = ( x[:, :, 0] - imagenet_stats["mean"][2] ) / imagenet_stats["std"][2]
        x[:, :, 1] = ( x[:, :, 1] - imagenet_stats["mean"][1] ) / imagenet_stats["std"][1]
        x[:, :, 2] = ( x[:, :, 2] - imagenet_stats["mean"][0] ) / imagenet_stats["std"][0]

        return x

class NormalizeGray_OCV_naive(object):
    def __init__(self, s, a):
        super(NormalizeGray_OCV_naive, self).__init__()
        self.s = s
        self.a = a

    def __call__(self, x):
        x = x / self.s - self.a
        return x.astype(np.float32)

class GrayscaleNoTensor(object):
    def __call__(self, x):
        return cv2.cvtColor( x, cv2.COLOR_BGR2GRAY )

class SobelXNoTensor(object):
    def __call__(self, x):
        return cv2.Sobel(x, cv2.CV_32FC1, 1, 0)

class ResizeNoTensor(object):
    def __init__(self, h, w):
        super(ResizeNoTensor, self).__init__()
        
        self.h = h # The new height.
        self.w = w # The new width.

    def __call__(self, x):
        # Assuming an OpenCV image.
        return cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

class ResizeDisparityNoTensor(object):
    def __init__(self, h, w):
        super(ResizeDisparityNoTensor, self).__init__()
        
        self.h = h # The new height.
        self.w = w # The new width.

    def __call__(self, x):
        # Assuming an OpenCV image with float data type.
        # The factor.
        f = self.w / x.shape[1]

        return cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_LINEAR) * f

class RandomCropSized_OCV(object):
    def __init__(self, h, w):
        super(RandomCropSized, self).__init__()

        self.h = h
        self.w = w

    def __call__(self, x):
        """
        x is an OpenCV mat.
        x must be larger than self.h and self.w
        """

        # Allowed indices.
        ah = x.shape[0] - self.h
        aw = x.shape[1] - self.w

        if ( ah < 0 ):
            raise Exception("x.shape[0] < self.h. x.shape[0] = {}, self.h = {}. ".format( x.shape[0], self.h ))

        if ( aw < 0 ):
            raise Exception("x.shape[1] < self.W. x.shape[1] = {}, self.W = {}. ".format( x.shape[1], self.W ))

        # Get two random numbers.
        ah = ah + 1
        aw = aw + 1

        ch = np.random.randint(0, ah)
        cw = np.random.randint(0, aw)

        return x[ch:ch+self.h, cw:cw+self.w]

class NormalizeSelf_OCV_01(object):
    def __call__(self, x):
        """
        x is an OpenCV mat.
        """

        x = x.astype(np.float32)

        x = x - x.min()
        x = x / x.max()

        return x

class NormalizeSelf_OCV(object):
    def __call__(self, x):
        """
        x is an OpenCV mat.
        This functiion perform normalization for individual channels.

        The normalized mat will have its values ranging from -1 to 1.
        """

        if ( 2 == len(x.shape) ):
            # Single channel mat.
            s = x.std()
            x = x - x.mean()
            x = x / s
        elif ( 3 == len(x.shape) ):
            # 3-channel mat.
            x = x.clone()

            for i in range(3):
                s = x[:, :, i].std()
                x[:, :, i] = x[:, :, i] - x[:, :, i].mean()
                x[:, :, i] = x[:, :, i] / s
        else:
            raise Exception("len(x.shape) = %d. ".format(len(x.shape)))

        return x

class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)

class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)

class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)

class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img

class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))
