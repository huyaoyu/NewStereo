#!/bin/bash

echo "Run ApplyRecurrentSinglePatch.sh."

# python ApplyRecurrentSinglePatch.py \
#     /home/yaoyu/Projects/NewStereo/Trial/RecurrentStereo/WD/InitTest/models/IT_02_RS_00.pkl \
#     /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im0.png \
#     /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im1.png \
#     ./WD/Recurrent_SmallSize \
#     "500, 650, 1011, 1161" \
#     --init-disp-width 480 \
#     --max-disp 16

python ApplyRecurrentSinglePatch.py \
    /home/yaoyu/Projects/NewStereo/Trial/RecurrentStereo/WD/InitTest/models/IT_02_RS_00.pkl \
    /home/yaoyu/temp/SceneFlowSample/FlyingThings3D/RGB_cleanpass/left/0007.png \
    /home/yaoyu/temp/SceneFlowSample/FlyingThings3D/RGB_cleanpass/right/0007.png \
    ./WD/Recurrent_SceneFlow \
    "200, 28, 711, 539" \
    --init-disp-width 512 \
    --max-disp 16

echo "Done with ApplyRecurrent.py."