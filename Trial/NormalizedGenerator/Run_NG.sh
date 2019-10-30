#!/bin/bash

echo "Run Run_NG.py"

python Run_NG.py --working-dir ./WD/Debug \
    --grayscale \
    --dl-batch-size 2 \
    --dl-num-workers 2 \
    --dl-crop-train "512, 512" \
    --dl-crop-test "512, 512" \
    --data-root-dir /home/yaoyu/expansion/StereoData/SceneFlowSample/FlyingThings3D \
    --data-file-list \
    --train-epochs 1000 \
    --test-loops 100 \
    --lr 0.001
    # --read-model NG.pkl

echo "Done with Run_NG.sh"
