from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np

from numba import cuda

from CommonPython.ImageMisc.ImageCheck import check_same_dimension

@cuda.jit
def k_warp_test_image_c1(img, disp, warped, invalid):
    # Get the size of the image.
    H = img.shape[0]
    W = img.shape[1]

    # Get the cuda kernel index.
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Get the step size in y.
    strideY = cuda.blockDim.y * cuda.gridDim.y

    for y in range(ty, H, strideY):
        for i in range(W):
            # Get the disparity.
            d = disp[y, i]

            if ( d <= invalid ):
                continue

            d = round( d )

            # The coordinate in the test image.
            xt = int(i - d)

            if (xt < 0):
                continue

            # Assign the value to the warped image.
            warped[y, i] = img[y, xt]

@cuda.jit
def k_warp_test_image_c3(img, disp, warped, invalid):
    # Get the size of the image.
    H = img.shape[0]
    W = img.shape[1]

    # Get the cuda kernel index.
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Get the step size in y.
    strideY = cuda.blockDim.y * cuda.gridDim.y

    for y in range(ty, H, strideY):
        for i in range(W):
            # Get the disparity.
            d = disp[y, i]

            if ( d <= invalid ):
                continue

            d = round( d )

            # The coordinate in the test image.
            xt = int(i - d)

            if (xt < 0):
                continue

            # Assign the values to the warped image.
            warped[y, i, 0] = img[y, xt, 0]
            warped[y, i, 1] = img[y, xt, 1]
            warped[y, i, 2] = img[y, xt, 2]

def warp_test_image(img, disp):
    """
    img: The test image.
    disp: The disparity map of the reference image.
    """

    # Check the size of the inputs.
    if ( not check_same_dimension((img, disp)) ):
        raise Exception("The dimensions of the input files are not compatible. img.shape = {}, disp.shape = {}. ".format( img.shape, disp.shape ))

    # Assume the dtype of img is float32.
    if ( np.float32 != img.dtype ):
        raise Exception("dtype of img should be float32. img.dtype = {}.".format(img.dtype))

    # Initialize a blank image.
    warped = np.zeros_like(img, dtype=np.float32) - 1.0

    dImg    = cuda.to_device(img)
    dDisp   = cuda.to_device(disp)
    dwarped = cuda.to_device(warped)

    if ( 2 == len( img.shape ) ):
        cuda.synchronize()
        k_warp_test_image_c1[[1, 100, 1], [1, 32, 1]](dImg, dDisp, dwarped, 192)
        cuda.synchronize()
    elif ( 3 == len( img.shape ) ):
        cuda.synchronize()
        k_warp_test_image_c3[[1, 100, 1], [1, 32, 1]](dImg, dDisp, dwarped, 192)
        cuda.synchronize()
    else:
        raise Exception("Only supports 1-channel and 3-channel image. len(img.shape) == {}. ".format(len(img.shape)))

    warped = dwarped.copy_to_host()

    # Filter all the negative values.
    mask = warped < 0
    warped[mask] = 0

    return np.clip( warped, 0.0, 255.0 ).astype(np.uint8)

if __name__ == "__main__":
    print("Test warp.")

    # Filenames.
    imgFn_0 = "/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/200122/Rectified_L_color.jpg"
    imgFn_1 = "/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/200122/Rectified_R_color.jpg"
    dispFn  = "/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/200122/Rectified_L_color_Disparity.npy"

    # Open the reference and test images.
    img_0 = cv2.imread(imgFn_0, cv2.IMREAD_UNCHANGED)
    img_1 = cv2.imread(imgFn_1, cv2.IMREAD_UNCHANGED)

    # Open the disparity map of the reference image.
    disp = np.load(dispFn).astype(np.float32)

    # Warp the test image.
    warped_10 = warp_test_image(img_1.astype(np.float32), disp)

    # Convert the image.
    warped_10 = np.clip(warped_10, 0, 255).astype(np.uint8)

    # Save the warped image.
    cv2.imwrite("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/200122/Rectified_R_color_warped.png", warped_10, \
        [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Make a grayscale image.
    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

    # Warp the test image.
    warped_10 = warp_test_image(gray.astype(np.float32), disp)

    # Convert the image.
    warped_10 = np.clip( warped_10, 0, 255 ).astype(np.uint8)

    # Save the warped image.
    cv2.imwrite("/home/yaoyu/Transient/CTBridgeGirder02Aug27Side01_05/200122/Rectified_R_color_warped_gray.png", warped_10, \
        [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print("Done.")