#!/bin/bash

# Train the model using the full dataset with a reduced batch size of 2
# This helps manage GPU memory usage for machines with limited VRAM
python train.py \
  --train_file data/processed/train.json \
  --val_file data/processed/val.json \
  --test_file data/processed/test.json \
  --device cuda \
  --run_name "full-dataset-run"

# Example of resuming training from a checkpoint (commented out by default)
# To resume training from the best model checkpoint, uncomment the following lines:
# python train.py \
#   --train_file data/processed/train.json \
#   --val_file data/processed/val.json \
#   --test_file data/processed/test.json \
#   --device cuda \
#   --run_name "resumed-training-run" \
#   --resume_from_checkpoint "best_model.pt" 