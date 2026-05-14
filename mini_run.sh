#!/bin/bash

python run_versa_quality.py \
    --csv          test.csv \
    --dataset_name test_enenlhet \
    --out_dir      test_enenlhet/ \
    --versa_repo   /home/prudhome/versa \
    --score_config ./audio_quality_config.yaml \
    --has_header \
    --no_text \
    --use_gpu
