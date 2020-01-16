#!/bin/bash

echo "Run Run_SSS.py"

python Run_SSS.py --working-dir ./WD/ZN_K1_ED \
    --grayscale \
    --dl-batch-size 2 \
    --dl-num-workers 2 \
    --dl-crop-train "0, 0" \
    --dl-crop-test "0, 0" \
    --dl-resize "256, 456" \
    --data-root-dir /home/yaoyu/temp/SceneFlowSample \
    --data-file-list \
    --train-epochs 1000 \
    --test-loops 100 \
    --lr 0.001 \
    # --read-model SSS.pkl \
    # --read-optimizer SSS_Opt.pkl

echo "Done with Run_SSS.sh"
