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