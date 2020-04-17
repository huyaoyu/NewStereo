#!/bin/bash

echo "Run ApplyRecurrentSinglePatch.sh."

# python ApplyRecurrentSinglePatch.py \
#     /home/yaoyu/Projects/NewStereo/Trial/OPT/WD/ClampsBender/models/IT_10_AutoSave_00009000_00.pkl \
#     /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im0.png \
#     /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im1.png \
#     ./WD/Recurrent_Patch_ClampsBender \
#     "500, 650, 1011, 1161" \
#     --init-disp-width 512 \
#     --max-disp 24

python ApplyRecurrentSinglePatch.py \
    /home/yaoyu/Projects/NewStereo/Trial/OPT/WD/StackNP/models/SNP_01_AutoSave_00006000_00.pkl \
    /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im0.png \
    /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im1.png \
    ./WD/Recurrent_Patch_StackNP \
    "500, 650, 1011, 1161" \
    --init-disp-width 512 \
    --max-disp 24

# python ApplyRecurrentSinglePatch.py \
#     /home/yaoyu/Projects/NewStereo/Trial/OPT/WD/SmallSize/models/SS_01_PWCNS_00.pkl \
#     /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im0.png \
#     /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im1.png \
#     ./WD/Recurrent_Patch_SmallSize \
#     "500, 650, 1011, 1161" \
#     --init-disp-width 512 \
#     --max-disp 16

echo "Done with ApplyRecurrentSinglePatch.py."