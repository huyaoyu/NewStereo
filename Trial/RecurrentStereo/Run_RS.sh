#!/bin/bash

echo "Run Run_RS.py"

python Run_RS.py --working-dir ./WD/InitTest \
    --grayscale \
    --max-disparity 16 \
    --dl-batch-size 1 \
    --dl-num-workers 1 \
    --dl-crop-train "0, 0" \
    --dl-crop-test "0, 0" \
    --dl-resize "288, 512" \
    --data-root-dir /home/yaoyu/expansion/StereoData/SceneFlowSample/FlyingThings3D \
    --data-file-list \
    --train-epochs 1000 \
    --test-loops 100 \
    --optimizer adam \
    --lr 0.0005 \
    --corr-k 3 \
    --flow-amp 1 \
    # --read-model RS.pkl \
    # --read-optimizer RS_Opt.pkl

echo "Done with Run_RS.sh"
