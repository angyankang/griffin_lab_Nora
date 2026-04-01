#! /usr/bin/bash
set -e

hf download \
    --repo-type dataset \
    --local-dir ./data/interndata-a1 \
    yixuan-tan/InternData-A1-LeRobot-v3.0-by-embodiment
