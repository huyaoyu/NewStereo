#!/bin/bash

echo "Run Run_PWCNS.py"

python Run_PWCNS.py --working-dir ./WD/InitTest \
    --grayscale \
    --max-disparity 24 \
    --dl-batch-size 1 \
    --dl-num-workers 1 \
    --dl-crop-train "0, 0" \
    --dl-crop-test "0, 0" \
    --dl-resize "512, 960" \
    --data-root-dir /home/yaoyu/temp/SceneFlowSample \
    --data-file-list \
    --train-epochs 1000 \
    --test-loops 100 \
    --optimizer adam \
    --lr 0.0005 \
    --corr-k 3 \
    --flow-amp 1 \
    # --read-model PWCNS.pkl \
    # --read-optimizer PWCNS_Opt.pkl

echo "Done with Run_PWCNS.sh"
