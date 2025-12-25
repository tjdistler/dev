# GPT-1 from Scratch

A from-scratch implementation of OpenAI's GPT-1 transformer model. This project implements all transformer components (attention, MLPs, layer normalization, etc.) manually while using PyTorch's basic tensor operations and data types.

## Project Goals

- Recreate the original GPT-1 architecture
- Implement all transformer components from scratch (attention, MLP, layer norm, etc.)
- Use only PyTorch's basic tensor types and math operations
- Learn the transformer architecture step by step

## Setup

1. Create and activate the virtual environment:
   ```bash
   ./activate_env.sh
   ```

   Or manually:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── README.md              # Project documentation
├── activate_env.sh        # Script to create and activate virtual environment
├── requirements.txt       # Python dependencies
├── gpt/                   # Main GPT package
│   ├── __init__.py
│   ├── config.py          # Model configuration
│   ├── layer_norm.py      # Layer normalization implementation
│   ├── mlp.py             # Multi-layer perceptron (feed-forward)
│   ├── attention.py       # Multi-head self-attention
│   ├── embeddings.py      # Token and positional embeddings
│   ├── transformer_block.py  # Transformer decoder block
│   └── model.py           # Main GPT model
├── tokenizer.py           # Tokenizer implementation
├── data.py                # Data loading and preprocessing
├── train.py               # Training script
├── generate.py            # Text generation script
└── tests/                 # Unit tests
    ├── __init__.py
    ├── conftest.py        # Shared pytest fixtures
    ├── test_config.py     # Tests for configuration
    ├── test_layer_norm.py # Tests for layer normalization
    ├── test_mlp.py        # Tests for MLP
    ├── test_attention.py  # Tests for attention
    ├── test_embeddings.py # Tests for embeddings
    ├── test_transformer_block.py  # Tests for transformer block
    ├── test_model.py      # Tests for GPT model
    ├── test_tokenizer.py  # Tests for tokenizer
    └── test_data.py       # Tests for data loading
```

## Implementation Status

This project is structured for step-by-step implementation. All components are scaffolded with placeholder classes and functions - ready for you to implement!

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest tests/ --cov=gpt --cov-report=html
```

## Usage

Once implemented, you'll be able to:
- Train the model: `python train.py`
- Generate text: `python generate.py`

