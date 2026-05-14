#!/bin/bash

python versa/bin/scorer.py \
    --score_config temp/tiny.yaml \
    --pred temp/tiny.scp \
    --gt None \
    --output_file temp/tiny.out \
    --io soundfile \
    --use_gpu True


cat temp/tiny.out
echo
echo


python versa/bin/scorer.py \
    --score_config temp/neural.yaml \
    --pred temp/tiny.scp \
    --gt None \
    --output_file temp/neural.out \
    --io soundfile \
    --use_gpu True


cat temp/neural.out


