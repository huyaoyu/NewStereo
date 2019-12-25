#!/bin/bash

echo "Run Run_SSS.py"

python Run_SSS.py --working-dir ./WD/Debug \
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
    # --read-model SSS.pkl

echo "Done with Run_SSS.sh"
