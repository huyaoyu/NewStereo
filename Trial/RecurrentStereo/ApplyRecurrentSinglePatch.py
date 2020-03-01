
import argparse
import cv2
import math
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Recurrent component.
from Model.RecurrentStereo import RecurrentStereo as RCom
from Model.RecurrentStereo import RecurrentStereoParams as RComParams
from Model.Recurrent import RecurrentModel as RModel
from Model.ImageStack import stack_single_channel_tensor

from Model.StereoUtility import WarpByDisparity

from CommonPython.Filesystem import Filesystem
from CommonPython.ImageWrite.ImageWrite import write_float_image_normalized, write_float_image_fixed_normalization

def extract_integers_from_argument(arg, expected=0):
    """
    arg is a string separated by commas.
    expected is the expected number to be extracted. Set 0 to disable.
    """

    ss = arg.split(",")

    ns = [ int(s.strip()) for s in ss ]

    if ( expected > 0 and len(ns) != expected ):
        raise Exception("Expecting {} integers. {} extracted from {}. ".format(expected, len(ns), arg))

    return ns

def read_image(fn):
    if ( not os.path.isfile(fn) ):
        raise Exception("%s does not exist. " % (fn))

    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)

    if ( 3 == img.ndim ):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

def normalize_image(img, s=255.0, a=0.5):
    # Convert the image to float.
    img = img.astype(np.float32)

    img = img / s - a

    return img

def read_disparity(fn):
    if ( not os.path.isfile(fn) ):
        raise Exception("%s does not exist. " % (fn))

    disp = np.load(fn)

    return disp

def find_closest_smaller_integer(val, base):
    base = int(base)
    assert base > 0
    assert val > base

    d = int(val) // base

    return int(d*base)

def dividable(val, base):
    assert val >= base
    assert base > 0

    if ( 0 == val % base ):
        return True
    else:
        return False

def load_pytorch_model(model, modelname):
        preTrainDict = torch.load(modelname)
        model_dict = model.state_dict()
        # print 'preTrainDict:',preTrainDict.keys()
        # print 'modelDict:',model_dict.keys()
        preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict}

        if( 0 == len(preTrainDictTemp) ):
            print("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]

                if ( kk in model_dict ):
                    preTrainDictTemp[kk] = v

            preTrainDict = preTrainDictTemp

        if ( 0 == len(preTrainDict) ):
            raise Exception("Could not load model from %s." % (modelname))

        # for item in preTrainDict:
        #     print("Load pretrained layer:{}".format(item) )

        model_dict.update(preTrainDict)
        model.load_state_dict(model_dict)
        return model

def create_and_load_recurrent_component(fn, maxDisp, corrK=3, amp=1.0, flagGray=True, flagMultiGPU=False):
    if ( not os.path.isfile(fn) ):
        raise Exception("Recurrent component {} does not exist. ".format( fn ))

    # Create the model.
    params = RComParams()
    params.set_max_disparity(maxDisp)
    params.corrKernelSize = corrK
    params.amp            = amp
    params.flagGray       = flagGray

    model = RCom(params)

    # Load the PyTorch model.
    model = load_pytorch_model(model, fn)

    print("The recurrent component has %d model parameters." % \
            ( sum( [ p.data.nelement() for p in model.parameters() ] ) ))

    if ( flagMultiGPU ):
        model = nn.DataParallel(model)

    model.cuda()

    return model

def convert_image_2_tensor(img):
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda()
    t.requires_grad = False

    return t

def convert_cuda_tensor_2_image(t):
    return t.squeeze(0).squeeze(0).detach().cpu().numpy()

def estimate_initial_disparity(img0, img1, model, initSize):
    """
    Size is in (H, W).
    """

    # Original image size.
    oldH, oldW = img0.shape

    # Resize the input image.
    newH, newW = initSize

    rImg0 = cv2.resize( img0, (newW, newH), interpolation=cv2.INTER_LINEAR )
    rImg1 = cv2.resize( img1, (newW, newH), interpolation=cv2.INTER_LINEAR )

    # Convert the images into torch tensors.
    t0 = convert_image_2_tensor(rImg0)
    t1 = convert_image_2_tensor(rImg1)

    # Create stacks.
    stack0 = stack_single_channel_tensor(t0, shift=8, radius=32)
    stack1 = stack_single_channel_tensor(t1, shift=8, radius=32)

    with torch.no_grad():
        # Forward.
        disp0, disp1, disp2, disp3 = model(stack0, stack1)

        print("disp0.szie() = {}".format(disp0.size()))

        # Upsample to the original size.
        upDisp0 = F.interpolate( disp0, (oldH, oldW), mode='bilinear', align_corners=False ) * (1.0 * oldW / newW)

    # Copy the data to CPU.
    disp0CPU   = convert_cuda_tensor_2_image(disp0)
    upDisp0CPU = convert_cuda_tensor_2_image(upDisp0)

    return upDisp0CPU, disp0CPU

def save_disp(fnBase, disp, dispMin=0, dispMax=256):
    outFn = "%s.npy" % (fnBase)
    np.save(outFn, disp)

    outFn = "%s.png" % (fnBase)
    write_float_image_fixed_normalization(outFn, disp, dispMin, dispMax)

def mark_rectangle_single_channel_float(img, coordinates, m0, m1):
    """
    img is a single channel image in float data type.
    coordinates is the coordinates of the two corners, (x0, y0, x1, y1)
    m0, m1: The bounds.
    """

    assert ( m1 > m0 ), "m1 ({}) must be larger than m0 ({}). ".format( m1, m0 )

    # Local copy of img.
    img = img.astype(np.float32)
    img = np.clip(img, m0, m1)
    img = ( img - m0 ) / ( m1 - m0 ) * 255
    img = img.astype(np.uint8)

    # Transform the img into a 3-channel, 0-255 valued image.
    tempImg = np.stack((img, img, img), axis=-1)

    # Draw the rectangle.
    tempImg = cv2.rectangle(tempImg, \
        (coordinates[0], coordinates[1]), \
        (coordinates[2], coordinates[3]), 
        (0, 0, 255), 1)

    return tempImg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Appy recurrent model on a single image.")

    parser.add_argument("rcom", type=str, \
        help="The recurrent component.")

    parser.add_argument("infile0", type=str, \
        help="The input file 0.")

    parser.add_argument("infile1", type=str, \
        help="The input file 1.")

    parser.add_argument("outdir", type=str, \
        help="The output directory.")

    parser.add_argument("coordinates", type=str, \
        help="A 4-number string separated by commas. x0, y0, x1, y1")

    parser.add_argument("--init-disp-width", type=int, default="512", \
        help="The initial disparity width.")

    parser.add_argument("--max-disp", type=int, \
        help="The max disparity in each pyramid level.")

    parser.add_argument("--corr-k", type=int, default=3, \
        help="Correlation kernel size.")
    
    parser.add_argument("--amp", type=float, default=1.0, \
        help="The amplitude parameter.")
    
    args = parser.parse_args()

    # Test outdir.
    Filesystem.test_directory(args.outdir)

    # Extract the coordinates.
    coor = extract_integers_from_argument(args.coordinates, 4)

    print("coor = {}. ".format(coor))

    # Open the two files.
    img0_Ori = read_image(args.infile0)
    img1_Ori = read_image(args.infile1)

    img0 = normalize_image(img0_Ori)
    img1 = normalize_image(img1_Ori)
    
    hOri, wOri = img0.shape
    print("Original image size is (H%d, W%d). " % ( hOri, wOri ))

    # The initial disparity size.
    initDispW = args.init_disp_width

    assert (initDispW > 0 and initDispW < wOri), "initDispW = {}. wOri = {}. ".format( initDispW, wOri )
    assert ( dividable( initDispW, 32 ) ), "The initial disparity wdith ({}) must be dividable by 32. ".format(initDispW)

    initDispH = int( 1.0 * initDispW / wOri * hOri )

    initDispH = find_closest_smaller_integer( initDispH, 32 )

    print("Initial disparity size is (H%d, W%d). " % ( initDispH, initDispW ))

    # Load the recurrent component.
    assert ( args.max_disp > 0 ), "Wrong --max-disp value: {}. ".format( args.max_disp )
    assert ( args.corr_k > 0 ), "Wrong --corr-k value: {}. ".format( args.corr_k )
    recCom = create_and_load_recurrent_component(args.rcom, \
        args.max_disp, args.corr_k, args.amp, flagGray=True, flagMultiGPU=False)

    # import ipdb; ipdb.set_trace()
    print("recCom.headFE.firstConv.model[1].weight[0, 0, :, :] = \n{}".format( \
        recCom.headFE.firstConv.model[1].weight[0, 0, :, :] ))

    recCom.eval()

    # Create the recurrent model.
    recModel = RModel(recCom)

    # Initial disparity prediction.
    upInitDisp, initDisp = estimate_initial_disparity( img0, img1, recCom, (initDispH, initDispW) )
    
    # Save the initial disparity prediction.
    outFnBase = "%s/UpInitDisp" % (args.outdir)
    save_disp(outFnBase, upInitDisp, dispMin=0, dispMax=192)

    outFnBase = "%s/initDisp" % (args.outdir)
    save_disp(outFnBase, initDisp, dispMin=0, dispMax=192)

    # Original images.
    print("Original images.")

    # Convert disparity to torch tensor.
    upInitDispTensor = convert_image_2_tensor(upInitDisp)

    # Convert the images into torch tensors.
    t0 = convert_image_2_tensor(img0)
    t1 = convert_image_2_tensor(img1)

    # Warp t1.
    warp = WarpByDisparity()

    with torch.no_grad():
        t1 = warp(t1, upInitDispTensor)

    # Create stacks.
    stack0 = stack_single_channel_tensor(t0, shift=16, radius=32)
    stack1 = stack_single_channel_tensor(t1, shift=16, radius=32)

    # Work on the specified image region.
    with torch.no_grad():
        disp0, extra = recModel.apply( stack0, stack1, upInitDispTensor, coor, flagWarp=False )

    # disp1 = extra[0]

    disp0CPU = convert_cuda_tensor_2_image(disp0)
    # disp1CPU = convert_cuda_tensor_2_image(disp1)
    # dispRes1CPU = convert_cuda_tensor_2_image(extra[1])
    # dispRe0CPU  = convert_cuda_tensor_2_image(extra[2])
    dispRe0CPU  = convert_cuda_tensor_2_image(extra[0])

    # Save the disparity.
    outFnBase = "%s/DispPatch_%d_%d_%d_%d" % (args.outdir, coor[0], coor[1], coor[2], coor[3])
    save_disp(outFnBase, disp0CPU, dispMin=0, dispMax=192)

    # Save a image with rectangle marker.
    markedDisp = mark_rectangle_single_channel_float( upInitDisp, coor, m0=0, m1=192 )
    outFn = "%s/UpInitDisp_M_%d_%d_%d_%d.png" % (args.outdir, coor[0], coor[1], coor[2], coor[3])
    cv2.imwrite(outFn, markedDisp, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # # Save the residual disparity.
    # outFnBase = "%s/DispPatch_%d_%d_%d_%d_dispRes1" % (args.outdir, coor[0], coor[1], coor[2], coor[3])
    # save_disp(outFnBase, dispRes1CPU * 10, dispMin=0, dispMax=192)

    # Save the refinement disparity.
    outFnBase = "%s/DispPatch_%d_%d_%d_%d_dispRe0" % (args.outdir, coor[0], coor[1], coor[2], coor[3])
    save_disp(outFnBase, dispRe0CPU * 10, dispMin=0, dispMax=192)

    print("Done.")