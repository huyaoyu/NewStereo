#!/bin/bash

echo "Run Run_RS.py"

python Run_RS.py --working-dir ./WD/InitTest \
    --grayscale \
    --max-disparity 4 \
    --dl-batch-size 2 \
    --dl-num-workers 2 \
    --dl-crop-train "0, 0" \
    --dl-crop-test "0, 0" \
    --dl-resize "256, 448" \
    --data-root-dir /home/yaoyu/expansion/StereoData/SceneFlowSample/FlyingThings3D \
    --data-file-list \
    --train-epochs 1000 \
    --test-loops 100 \
    --lr 0.00001 \
    --read-model RS.pkl \
    --read-optimizer RS_Opt.pkl

echo "Done with Run_RS.sh"
