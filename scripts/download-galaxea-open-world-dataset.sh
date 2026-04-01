#! /usr/bin/bash
set -e

# don't download subsets yet
hf download \
    --repo-type dataset \
    --local-dir ./data/galaxea-open-world-dataset \
    griffinlabs/Galaxea-Open-World-Dataset-LeRobot-v3.0 \
    --exclude "subsets/"

# download select subsets
hf download \
    --repo-type dataset \
    --local-dir ./data/galaxea-open-world-dataset \
    griffinlabs/Galaxea-Open-World-Dataset-LeRobot-v3.0 \
    --include \
        $(sed 's/$/\//' < arm-and-gripper-only-subsets.txt)

# move delta norm stats to correct location
mv \
    ./data/galaxea-open-world-dataset/arm_and_gripper_only_delta_norm_stats.json \
    ./data/galaxea-open-world-dataset/delta_norm_stats.json
