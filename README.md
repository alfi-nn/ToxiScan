# Bio-ChemTransformer

A sequence-to-sequence transformer model for predicting Adverse Drug Reactions (ADRs) using biomedical text and chemical structure information.

## Overview

Bio-ChemTransformer is an encoder-decoder transformer architecture that combines:
- Bio_ClinicalBERT for biomedical text embeddings
- ChemBERT for molecular structure (SMILES) embeddings
- Diagonal-Masked Attention to prevent information leakage
- Transformer decoder with cross-attention for ADR prediction

## Project Structure

```
biochempro/
├── data/                      # Data storage and preprocessing
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed datasets
│   └── preprocessing.py       # Data preprocessing scripts
├── models/                    # Model implementation
│   ├── embedding.py           # Embedding layer implementation
│   ├── encoder.py             # Encoder with DMA implementation
│   ├── decoder.py             # Decoder implementation
│   └── transformer.py         # Complete transformer model
├── training/                  # Training utilities
│   ├── trainer.py             # Training loop implementation
│   └── metrics.py             # Evaluation metrics
├── utils/                     # Utility functions
│   ├── tokenizers.py          # Tokenization utilities
│   └── data_utils.py          # Data handling utilities
├── config.py                  # Configuration parameters
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script
├── inference.py               # Inference script
└── requirements.txt           # Project dependencies
```

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Linux/Mac: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Data

The model requires two types of data:
1. ADR text descriptions (for Bio_ClinicalBERT)
2. SMILES strings (for ChemBERT)

These can be sourced from databases like DailyMed, SIDER, or FAERS.

## Usage

### Training

```bash
python train.py --config configs/default.yaml
```

### Evaluation

```bash
python evaluate.py --model_path checkpoints/model.pt --test_data data/processed/test.json
```

### Inference

```bash
python inference.py --model_path checkpoints/model.pt --drug "Drug name" --smiles "SMILES_string"
```

## License

[MIT License](LICENSE) 