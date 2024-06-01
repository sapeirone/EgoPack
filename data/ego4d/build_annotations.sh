#!/bin/bash

echo "USAGE ./build_annotations.sh <ego4d_root>"
echo "EXAMPLE ./build_annotations.sh /storage/ego4d/v2/"

ego4d_root=$1
echo "ego4d_root: $ego4d_root"

# Directory structure
# - raw/annotations should point to the directory containing the annotations (.json files)
# - raw/features/features/omnivore_image_swinl should point to the omnivore image features
# - raw/features/features/omnivore_video_swinl should point to the omnivore video features
# - raw/features/features/slowfast8x8_r101_k400 should point to the SlowFast (RN101) features

mkdir -p raw/annotations
mkdir -p raw/features

ln -s $ego4d_root/annotations $(pwd)/raw/
ln -s $ego4d_root/omnivore_image_swinl $(pwd)/raw/features
ln -s $ego4d_root/omnivore_video_swinl $(pwd)/raw/features
ln -s $ego4d_root/slowfast8x8_r101_k400 $(pwd)/raw/features
