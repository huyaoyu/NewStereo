
import argparse
import cv2
import math
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Recurrent component.
from Model.PWCNetStereo import PWCNetStereoRes as RCom
from Model.PWCNetStereo import PWCNetStereoParams as RComParams
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

    with torch.no_grad():
        # Create stacks.
        stack0 = stack_single_channel_tensor(t0, shift=16, radius=32)
        stack1 = stack_single_channel_tensor(t1, shift=16, radius=32)

        # Forward.
        disp0, disp1, disp2, disp3 = model(stack0, stack1, torch.zeros((1)), torch.zeros((1)))

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

def left_pad_tensor(t, patchSize):
    """
    t the 4D tensor.
    patchSize, a 2-element list/tuple/array, the patch size, (H, W).
    """

    pH, pW = patchSize

    assert ( pH > 0 and pW > 0 ), "Wrong patchSize ({}). ".format(patchSize)

    # Get the size of the tensor.
    B, C, tH, tW = t.size()

    assert ( tH > pH and tW > pW ), "Wrong tensor size ({}, {}) and patchSize ({}). ".format( \
        tH, tW, patchSize )

    # Calculate padding size.
    nH = int( np.ceil( 1.0 * tH / pH ) ) * pH
    nW = int( np.ceil( 1.0 * tW / pW ) ) * pW

    # Create new tensor.
    newT = torch.zeros( ( B, C, nH, nW ) ).float()
    if ( t.is_cuda ):
        newT = newT.cuda()

    # Copy values.
    newT[:, :, 0:tH , (nW-tW):] = t

    return newT

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

    parser.add_argument("--patch-size", type=str, default="512, 512", \
        help="The default silding patch size (H, W).")

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
    patchSize = extract_integers_from_argument(args.patch_size, 2)

    print("patchSize = {}. ".format(patchSize))

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
        args.max_disp, args.corr_k, args.amp, flagGray=True)

    recCom.eval()

    # Create the recurrent model.
    recModel = RModel(recCom)

    # Initial disparity prediction.
    upInitDisp, initDisp = estimate_initial_disparity( img0, img1, recCom, (initDispH, initDispW) )
    
    # Save the initial disparity prediction.
    outFnBase = "%s/UpInitDisp" % (args.outdir)
    save_disp(outFnBase, upInitDisp, dispMin=0, dispMax=1024)

    outFnBase = "%s/initDisp" % (args.outdir)
    save_disp(outFnBase, initDisp, dispMin=0, dispMax=1024)

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

    # Left-padding.
    t0 = left_pad_tensor(t0, patchSize)
    t1 = left_pad_tensor(t1, patchSize)
    t0.requires_grad = False
    t1.requires_grad = False

    upInitDispTensor = left_pad_tensor(upInitDispTensor, patchSize)
    upInitDispTensor.requires_grad = False

    # Loop over the input image area.
    tH = t0.size()[2]
    tW = t0.size()[3]
    pH, pW = patchSize
    nRow = int( tH / patchSize[0] )
    nCol = int( tW / patchSize[1] )

    fusedDisp = torch.zeros( (1, 1, tH, tW ) ).float().cuda()
    fusedDisp.requires_grad = False

    for i in range(nRow):
        y0 = i*pH
        y1 = y0 + pH - 1

        # Create stacks.
        with torch.no_grad():
            stack0 = stack_single_channel_tensor(\
                t0[:, :, y0:y1+1,:], shift=16, radius=32)
            stack1 = stack_single_channel_tensor(\
                t1[:, :, y0:y1+1,:], shift=16, radius=32)
            stack0.requires_grad = False
            stack1.requires_grad = False

        for j in range(nCol):
            x0 = j*pW
            x1 = x0 + pW - 1
            
            print("(%d, %d) of (%d, %d)" % (i, j, nRow, nCol))

            # Work on the specified image region.
            with torch.no_grad():
                disp0, extra = recModel.apply( \
                    stack0, stack1, upInitDispTensor[:,:,y0:y1+1,:], \
                    (x0, 0, x1, pH-1), flagWarp=False )

                # Copy result.
                fusedDisp[:, :, y0:y1+1, x0:x1+1] = disp0

    # Save the disparity.
    fusedDispCPU = fusedDisp.detach().squeeze(0).squeeze(0).cpu().numpy()
    outFnBase = "%s/FusedDisp" % (args.outdir)
    save_disp(outFnBase, fusedDispCPU, dispMin=0, dispMax=1024)

    outFnBase = "%s/FusedDispCropped" % (args.outdir)
    save_disp(outFnBase, fusedDispCPU[0:hOri, tW-wOri:], dispMin=0, dispMax=1024)

    print("Done.")