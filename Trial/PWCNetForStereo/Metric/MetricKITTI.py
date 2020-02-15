
import cv2
import numpy as np

def apply_metrics(dispT, dispP, mask=None, resize=None):
    """
    dispT: The true disparity. A 3D NumPy array. (B, H, W) where B is the minibach number.
    mask: The mask array. A 3D NumPy array. 1-element of mask will be used in metrics computation.
    dispP: The predicted disparity. A 3D Numpy array. (B, H, W) where B is the minibach number.
    resize: If it is not None, it should be a 2-element tuple or list. The order of the numbers is
    ( rH, rW ), meaning the new height and width to be used by resizing dispT. The true disparity value 
    will also be scaled. rH and rW must both be positive numbers.

    This function returns a series of metrics. Namely, the 1-pixel, 2-pixel, 3-pixel, 4-pixel, and 5-pixel
    thresholded average error in ratio with respect to the total valid pixel number. 
    And the average end point error measured as pixel/pixpel. 
    The returned vaues are arranged in a Bx6 array, where B is the minibatch size.
    """

    B = dispT.shape[0]

    # Check ths size of the input arrays.
    nDimDispT = len( dispT.shape )
    nDimDispP = len( dispP.shape )

    if ( 3 != nDimDispT or 3 != nDimDispP ):
        raise Exception( "nDimDispT = {}, nDimDispP = {}. ".format( nDimDispT, nDimDispP ) )

    if ( resize is not None ):
        if ( resize[0] <= 0 or resize[1] <= 0 ):
            raise Exception( "resize = {}. ".format( resize ) )

    # Resize.
    if ( resize is not None ):
        tempDispT = np.zeros(( B, resize[0], resize[1] ), dtype=np.float32)

        for i in range( B ):
            tempDispT[i, :, :] = cv2.resize( dispT[i, :, :], ( resize[1], resize[0] ), interpolation=cv2.INTER_LINEAR )
        
        dispT = tempDispT * resize[1] / dispT.shape[2]

    # Reshape the inputs.
    dispT = dispT.reshape( ( -1, dispT.shape[1]*dispT.shape[2] ) )
    dispP = dispP.reshape( ( -1, dispP.shape[1]*dispP.shape[2] ) )

    # Total number of valid pixels.
    N = dispT.shape[1]

    # Apply mask.
    if ( mask is not None ):
        mask  = mask.reshape( ( -1, mask.shape[1]*mask.shape[2], ) )
        mask  = mask == 1
        N     = np.sum( mask, axis=1 ).reshape((-1,1))
    
    # Compute difference.
    diff = np.abs( dispT - dispP )

    # Initialize the metrics.
    metrics = np.zeros( (B, 6), dtype=np.float32 )

    # 1-pixel.
    maskDiff = diff > 1
    maskDiff = maskDiff * mask
    metrics[:, 0] = maskDiff.sum( axis=1 ).astype(np.float32) / N

    # 2-pixel.
    maskDiff = diff > 2
    maskDiff = maskDiff * mask
    metrics[:, 1] = maskDiff.sum( axis=1 ).astype(np.float32) / N

    # 3-pixel.
    maskDiff = diff > 3
    maskDiff = maskDiff * mask
    metrics[:, 2] = maskDiff.sum( axis=1 ).astype(np.float32) / N

    # 4-pixel.
    maskDiff = diff > 4
    maskDiff = maskDiff * mask
    metrics[:, 3] = maskDiff.sum( axis=1 ).astype(np.float32) / N

    # 5-pixel.
    maskDiff = diff > 5
    maskDiff = maskDiff * mask
    metrics[:, 4] = maskDiff.sum( axis=1 ).astype(np.float32) / N
    
    # End point error.
    diff = diff * mask
    metrics[:, 5] = diff.sum( axis=1 ).astype(np.float32) / N

    return metrics




