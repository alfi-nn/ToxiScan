#!/bin/bash

# Train the model using the full dataset
python train.py \
  --train_file data/processed/train.json \
  --val_file data/processed/val.json \
  --test_file data/processed/test.json \
  --device cuda \
  --run_name "full-dataset-run" 