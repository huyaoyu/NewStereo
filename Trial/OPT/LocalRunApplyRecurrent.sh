#!/bin/bash

echo "Run ApplyRecurrent.sh."

# python ApplyRecurrent.py \
#     /home/yaoyu/Projects/NewStereo/Trial/PRU/WD/NormalLoss/models/PRU_01_PWCNS_00.pkl \
#     /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im0.png \
#     /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im1.png \
#     ./WD/Recurrent_Fused_NormalLoss \
#     --patch-size "512, 512" \
#     --init-disp-width 512 \
#     --max-disp 24

# python ApplyRecurrent.py \
#     /home/yaoyu/Projects/NewStereo/Trial/PRU/WD/NormalLoss/models/PRU_01_PWCNS_00.pkl \
#     /home/yaoyu/Projects/PPSR_StandAlone/Data/Bridge_0501_07/Rectified_L_color.jpg \
#     /home/yaoyu/Projects/PPSR_StandAlone/Data/Bridge_0501_07/Rectified_R_color.jpg \
#     ./WD/Recurrent_Fused_NormalLoss_4K \
#     --patch-size "512, 512" \
#     --init-disp-width 1024 \
#     --max-disp 24

# CASE=Bridge_0501_07
# python ApplyRecurrent.py \
#     /home/yaoyu/Projects/NewStereo/Trial/OPT/WD/ClampsBender/models/IT_10_AutoSave_00009000_00.pkl \
#     /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_L_color.jpg \
#     /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_R_color.jpg \
#     ./WD/Recurrent_Fused_ClampsBender_${CASE} \
#     --patch-size "512, 512" \
#     --init-disp-width 1024 \
#     --max-disp 24

# python ApplyRecurrent.py \
#     /home/yaoyu/Projects/NewStereo/Trial/PRU/WD/NormalLoss/models/PRU_01_PWCNS_00.pkl \
#     /home/yaoyu/Projects/PPSR_StandAlone/Data/ShimizuBeam08/Rectified_L_color.jpg \
#     /home/yaoyu/Projects/PPSR_StandAlone/Data/ShimizuBeam08/Rectified_R_color.jpg \
#     ./WD/Recurrent_Fused_NormalLoss_4K_ShimizuBeam08 \
#     --patch-size "512, 512" \
#     --init-disp-width 1024 \
#     --max-disp 24

# CASE=JapanWall08
# python ApplyRecurrent.py \
#     /home/yaoyu/Projects/NewStereo/Trial/OPT/WD/ClampsBender/models/IT_10_AutoSave_00009000_00.pkl \
#     /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_L_color.jpg \
#     /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_R_color.jpg \
#     ./WD/Recurrent_Fused_ClampsBender_${CASE} \
#     --patch-size "512, 512" \
#     --init-disp-width 1024 \
#     --max-disp 24

# CASE=SmithHall_0803_0269
# python ApplyRecurrent.py \
#     /home/yaoyu/Projects/NewStereo/Trial/OPT/WD/ClampsBender/models/IT_10_AutoSave_00009000_00.pkl \
#     /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_L_color.jpg \
#     /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_R_color.jpg \
#     ./WD/Recurrent_Fused_ClampsBender_${CASE} \
#     --patch-size "512, 512" \
#     --init-disp-width 1024 \
#     --max-disp 24

# CASE=SmithHall_0803_0269
# python ApplyRecurrent.py \
#     /home/yaoyu/Projects/NewStereo/Trial/OPT/WD/StackNP/models/SNP_01_AutoSave_00006000_00.pkl \
#     /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_L_color.jpg \
#     /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_R_color.jpg \
#     ./WD/Recurrent_Fused_StackNP_${CASE} \
#     --patch-size "512, 512" \
#     --init-disp-width 1024 \
#     --max-disp 24

# CASE=CFAPillar_0516_26
# python ApplyRecurrent.py \
#     /home/yaoyu/Projects/NewStereo/Trial/OPT/WD/ClampsBender/models/IT_10_AutoSave_00009000_00.pkl \
#     /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_L_color.jpg \
#     /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_R_color.jpg \
#     ./WD/Recurrent_Fused_ClampsBender_${CASE} \
#     --patch-size "512, 512" \
#     --init-disp-width 1024 \
#     --max-disp 24

CASE=CFAPillar_0516_26
python ApplyRecurrent.py \
    /home/yaoyu/Projects/NewStereo/Trial/OPT/WD/StackNP/models/SNP_01_AutoSave_00006000_00.pkl \
    /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_L_color.jpg \
    /media/yaoyu/DiskE/Projects/Data_PPSR_StandAlone/${CASE}/Rectified_R_color.jpg \
    ./WD/Recurrent_Fused_StackNP_${CASE} \
    --patch-size "512, 512" \
    --init-disp-width 1024 \
    --max-disp 24

# python ApplyRecurrent.py \
#     /home/yaoyu/Projects/NewStereo/Trial/PRU/WD/SmallSize/models/SS_01_PWCNS_00.pkl \
#     /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im0.png \
#     /media/yaoyu/DiskE/StereoData/MiddleburyDataSets/stereo/MiddEval3/trainingF/Teddy/im1.png \
#     ./WD/Recurrent_Fused_SmallSize \
#     --patch-size "512, 512" \
#     --init-disp-width 512 \
#     --max-disp 16

echo "Done with ApplyRecurrent.py."