# # concat flow 
# CUDA_VISIBLE_DEVICES=3 python demo.py \
#     --model models/raft-things.pth \
#     --path /data/video_restoration/REDS/train/train_blur/000  \
#     --save_path /data/video_restoration/REDS/optical_flow_results/raft_concat/train/train_blur/000 \
#     --concat_flow \


CUDA_VISIBLE_DEVICES=3 python demo.py \
    --model models/raft-things.pth \
    --path /data/video_restoration/Vids4/walk  \
    --save_path /data/video_restoration/Vids4/optical_flow_results/raft_concat/walk \
    --concat_flow 

