#!/bin/bash

cd ..

python test_aad.py \
    --checkpoint /RAID5/projects/likeyang/happy/MEGConformer/test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20260405_152418/best_model.pt --aad_n_trials 1000 \
    --aad_n_distractors 9