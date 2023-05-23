#!/bin/bash

python3 tools/dataset_converters/images2coco.py \
  data/coco/test2017 data/coco/classes.txt \
  image_info_test-dev2017.json

python3 tools/train.py \
  configs/self/rtmdet_tiny.py

python3 tools/test.py \
  configs/self/rtmdet_tiny.py \
  work_dirs/rtmdet_tiny/epoch_5.pth \
  --json-prefix tmp-test
