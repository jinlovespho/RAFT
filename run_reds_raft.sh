#!/bin/bash


for folder in /data/video_restoration/REDS4/train/train_blur/*; do 
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")  # this gives "001", "002", etc.
        echo "Processing folder: $folder"
        echo "folder name: $folder_name"

        CUDA_VISIBLE_DEVICES=3 python demo.py \
            --model models/raft-things.pth \
            --path "$folder" \
            --save_path /data/video_restoration/REDS4/optical_flow/raft/train/train_blur/$folder_name
        
        echo "Finished processing folder: $folder_name"
    fi
done

