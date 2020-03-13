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

    # Argument check.
    assert shift >= 0
    assert radius >= 0

    # Force the arguments to be integers.
    shift  = int(shift)
    radius = int(radius)

    # Dimensions.
    B, C, H, W = img.size()

    assert C == 1
    
    padRadius = radius * shift
    paddedH   = H # Only pad the x-axis.
    paddedW   = W + 2*padRadius

    # Create a temporary tensor.
    paddedImg = torch.zeros( ( B, 1, paddedH, paddedW ), dtype=torch.float32 )
    paddedImg.requires_grad = False

    if ( img.is_cuda ):
        paddedImg = paddedImg.cuda()

    # Create the output tensor.
    C = 2*radius + 1
    out = torch.zeros( (B, C, H, W), dtype=torch.float32 )

    # Temporarily disable the gradient.
    out.requires_grad = False

    if ( img.is_cuda ):
        out = out.cuda()

    # with torch.no_grad():
    # Copy data from img to paddedImg.
    paddedImg[:, :, :, padRadius:padRadius+W] = img.clone()

    # Copy data to the output tensor.
    for i in range(C):
        idxH = 0
        idxW = i * shift

        # Get the channel.
        out[:, i, :, :] = paddedImg[:, 0, idxH:idxH+H, idxW:idxW+W]

    # Enable the gradient.
    out.requires_grad = True

    return out

if __name__ == "__main__":
    print("Test ImageStack.py")

    row = torch.arange(1, 33)

    img = row.repeat((2,1,32,1))
    img[1, 0, :, :] = img[1, 0, :, :] + 100
    # import ipdb; ipdb.set_trace()

    print("img.size() = {}. ".format(img.size()))
    print("img[0, 0, :, :] = \n{}".format( img[0,0,:,:] ))
    print("img[1, 0, :, :] = \n{}".format( img[1,0,:,:] ))

    stacked = stack_single_channel_tensor(img, shift=2, radius=8)
    print("stacked.size() = {}. ".format(stacked.size()))

    print("stacked[0, 0, :, :]  = \n{}".format( stacked[0, 0, :, :] ))
    print("stacked[0, 8, :, :]  = \n{}".format( stacked[0, 8, :, :] ))
    print("stacked[0, 16, :, :] = \n{}".format( stacked[0, 16, :, :] ))

    print("stacked[1, 0, :, :]  = \n{}".format( stacked[1, 0, :, :] ))
    print("stacked[1, 8, :, :]  = \n{}".format( stacked[1, 8, :, :] ))
    print("stacked[1, 16, :, :] = \n{}".format( stacked[1, 16, :, :] ))
    