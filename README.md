# Bio-ChemTransformer

A sequence-to-sequence transformer model for predicting Adverse Drug Reactions (ADRs) using biomedical text and chemical structure information. This implementation integrates ChemBERT for improved molecular representation.

## Overview

Bio-ChemTransformer is an encoder-decoder transformer architecture that combines:
- Bio_ClinicalBERT for biomedical text embeddings
- ChemBERT for molecular structure (SMILES) embeddings
- Diagonal-Masked Attention to prevent information leakage
- Transformer decoder with cross-attention for ADR prediction

## Project Structure

```
bio-chemtransformer/
├── data/                      # Data storage and preprocessing
│   ├── raw/                   # Raw data files
│   │   ├── sider/             # SIDER dataset
│   │   ├── faers/             # FAERS dataset
│   │   └── dailymed/          # DailyMed dataset
│   ├── processed/             # Processed datasets
│   ├── convert_raw_data.py    # Convert raw data to standard format
│   ├── download_dailymed.py   # Download DailyMed data
│   ├── download_faers.py      # Download FAERS data
│   ├── download_structures.py # Download molecular structures
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── process_all_data.py    # Process all datasets
│   └── prepare_training_data.py # Split data for training
├── models/                    # Model implementation
│   ├── embedding.py           # Embedding layer with ChemBERT
│   ├── encoder.py             # Encoder with DMA implementation
│   ├── decoder.py             # Decoder implementation
│   └── transformer.py         # Complete transformer model
├── training/                  # Training utilities
│   ├── trainer.py             # Training loop implementation
│   └── metrics.py             # Evaluation metrics
├── utils/                     # Utility functions
│   ├── smiles_tokenization.py # SMILES tokenization utilities
│   └── data_utils.py          # Data handling utilities
├── config.py                  # Configuration parameters
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script
├── inference.py               # Inference script
└── requirements.txt           # Project dependencies
```

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

### 1. Download and Prepare the Required Datasets

#### SIDER Dataset
The SIDER dataset contains drug side effects and indications:

```bash
# Download SIDER drug structures (SMILES strings)
python data/download_structures.py
```

#### DailyMed Dataset
The DailyMed dataset contains labeled drug information including adverse reactions:

```bash
# Download adverse reactions from DailyMed
python data/download_dailymed.py
```

#### FAERS Dataset (Optional)
The FAERS dataset contains real-world adverse event reports:

```bash
# Download FAERS data (optional)
python data/download_faers.py
```

### 2. Process All Datasets

Process all downloaded datasets and combine them:

```bash
# Process all datasets
python data/process_all_data.py
```

### 3. Prepare Training Data

Split the processed data into train, validation, and test sets:

```bash
# Prepare training data
python data/prepare_training_data.py
```

## Training

Train the Bio-ChemTransformer model:

```bash
# Train the model with default settings
python train.py

# Train with custom settings
python train.py --train_file data/processed/train.json --val_file data/processed/val.json --test_file data/processed/test.json --device cuda --use_wandb
```

### Training with the Full Dataset

For training with the full dataset, we've provided a convenience script:

```bash
# Train with the full dataset
./train_full.sh
```

### Memory-Efficient Training

The model is configured to use a small batch size (2) with gradient accumulation (16 steps) to enable training on GPUs with limited VRAM while maintaining the statistical benefits of larger batch sizes.

To further adjust memory usage, you can modify:
- `per_device_train_batch_size` in `config.py` (smaller values use less memory)
- `gradient_accumulation_steps` in `config.py` (larger values compensate for smaller batch sizes)

### Checkpoint System

The training automatically saves checkpoints to the `./checkpoints` directory:

- **Best Model**: Saved whenever validation loss improves (`best_model.pt`)
- **Periodic Checkpoints**: Saved at regular intervals (`epoch_X.pt`)

Checkpoints include:
- Model weights
- Optimizer state
- Learning rate scheduler state

### Resuming Training from Checkpoints

You can resume training from a saved checkpoint:

```bash
# Resume from the best model
python train.py --resume_from_checkpoint checkpoints/best_model.pt --device cuda

# Resume from a specific epoch
python train.py --resume_from_checkpoint checkpoints/epoch_5.pt --device cuda
```

This is useful when:
- Training was interrupted and needs to be continued
- You want to train for additional epochs after initial training
- Fine-tuning an existing model with different hyperparameters

The checkpoint path can be absolute, relative, or just the filename (in which case it looks in the checkpoint directory).

## Evaluation

Evaluate the trained model on a test dataset:

```bash
python evaluate.py --model_path checkpoints/best_model.pt --test_file data/processed/test.json
```

## Inference

Use the trained model for inference:

```bash
python inference.py --model_path checkpoints/best_model.pt --drug_name "Aspirin" --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

## Testing with Custom Input Data

The `test.py` script provides an easy way to test the trained model with your own input data:

### Single Input Testing

Test the model with a single ADR text and SMILES string:

```bash
python test.py --checkpoint checkpoints/best_model.pt --mode single \
  --adr_text "The patient experienced headache" \
  --smiles "CC(=O)Nc1ccc(O)cc1"
```

### Batch Testing

Test the model with multiple inputs from a file:

```bash
python test.py --checkpoint checkpoints/best_model.pt --mode batch \
  --data_file example_test_data.jsonl \
  --output_file results.json
```

### Input Data Format

For batch testing, prepare your data in JSONL format (one JSON object per line):

```json
{"drug_name": "Acetaminophen", "smiles": "CC(=O)Nc1ccc(O)cc1", "adr_text": "The patient experienced headache and nausea after taking medication.", "source": "example"}
{"drug_name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "adr_text": "The patient reported stomach pain and dizziness.", "source": "example"}
```

Each object should contain:
- `drug_name`: Name of the drug (optional)
- `smiles`: SMILES notation for the drug's chemical structure
- `adr_text`: The ADR (Adverse Drug Reaction) text
- `source`: Source of the data (optional)

### Additional Options

The testing script supports several options:
- `--batch_size`: Number of samples to process at once (default: 8)
- `--device`: Device to run inference on (default: cuda if available, otherwise cpu)
- `--num_return_sequences`: Number of different outputs to generate for each input (in single mode)

## Model Architecture Details

### ChemBERT Integration

The Bio-ChemTransformer uses ChemBERT for molecular structure representation:

- **ChemBERT**: A pre-trained BERT model specialized on SMILES strings, offering improved molecular representation compared to general language models.
- **SMILES Tokenization**: SMILES strings are tokenized using ChemBERT's specialized tokenizer.
- **Embedding Combination**: Embeddings from Bio_ClinicalBERT (for ADR text) and ChemBERT (for molecules) are combined using either concatenation or summation.

### Diagonal-Masked Attention (DMA)

The model uses Diagonal-Masked Attention in the encoder to prevent information leakage:

- During training, random tokens in the ADR text are masked.
- The model learns to predict these masked tokens using both the surrounding text and molecular structure information.

## Acknowledgments

- The ChemBERTa model was developed by Seyonec and is available on Hugging Face: [ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
- Bio_ClinicalBERT was developed by Emily Alsentzer et al. and is available on Hugging Face: [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- SIDER dataset: [SIDER](http://sideeffects.embl.de/)
- DailyMed: [DailyMed](https://dailymed.nlm.nih.gov/dailymed/)
- FAERS: [FDA Adverse Event Reporting System](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-latest-quarterly-data-files)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 