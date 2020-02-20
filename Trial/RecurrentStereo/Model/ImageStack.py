import torch


def stack_single_channel_tensor(img, shift=16, radius=32):
    """
    img is a tensor which is obtained by coverting a signle channel NumPy 
    image. The dimension of img is (B, 1, H, W).

    The returned tensor has dimention (B, 2*radius+1, H, W).

    The returned tensor is a collection of the spatial shifts of the original image 
    with the original image being the (radius+1)th channel. The channels before the 
    original image are images shifted to the right. The channels after the original 
    image are those shifted to the left.

    The shifting step is defined by the value of the shift argument. The original image
    is firstly padded by radius*shift pixels and the padded value is zero. And the shifted
    images are sliced from the padded image with the size of the original image (H, W).
    """