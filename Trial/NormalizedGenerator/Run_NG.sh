#!/bin/bash

echo "Run Run_NG.py"

python Run_NG.py --working-dir ./WD/Dummy --grayscale --dl-batch-size 2 --dl-num-workers 2 --dl-crop-train "512, 512" --dl-crop-test "512, 512" --data-root-dir /home/yaoyu/expansion/StereoData/SceneFlowSample/FlyingThings3D --data-file-list --train-epochs 2

echo "Done with Run_NG.sh"
