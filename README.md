# LLM-Training-JAX

Build and train a Large Language Model (LLM) from scratch using JAX and Flax, covering data processing, transformer architecture, training loops, and optimization. This repository provides a hands-on implementation of a mini GPT-style model for story generation.

## Overview

This project demonstrates the complete pipeline for creating and training a language model using modern JAX/Flax libraries:

- **Data Processing**: Load and tokenize text data using the GPT-2 tokenizer
- **Model Architecture**: Build transformer blocks with multi-head attention and feed-forward layers
- **Training**: Implement training loops with optimizers and learning rate schedules
- **Inference**: Load checkpoints and generate text with trained models

## Key Libraries

- **JAX**: High-performance numerical computing library for ML
- **Flax NNX**: Neural network library for JAX with module-based API
- **Grain**: Efficient data loading pipeline
- **Optax**: Optimization library with learning rate schedules
- **Orbax**: Checkpoint management and serialization
- **Tiktoken**: OpenAI's BPE tokenizer

## Project Structure

### Notebooks

1. **Building the LLM Architecture** (`Building the LLM Architecture.ipynb`)
   - Token and position embeddings
   - Causal attention masking for autoregressive generation
   - Transformer blocks with multi-head attention and feed-forward networks
   - Complete model assembly

2. **Data Loading with Grain** (`Data Loading with Grain.ipynb`)
   - Load text data (TinyStories dataset)
   - Tokenization with GPT-2 tokenizer
   - Create efficient data loaders using Grain
   - Handle sequence padding and batching

3. **Training the Model** (`Training the Model.ipynb`)
   - Define loss functions (cross-entropy with integer labels)
   - Configure optimizers and learning rate schedules (warmup + cosine decay)
   - Training loop with gradient descent
   - Save checkpoints using Orbax

4. **Loading and Running a Pre-trained LLM** (`Loading and Running a Pre-trained LLM.ipynb`)
   - Load saved model checkpoints
   - Configure device sharding for CPU/GPU
   - Generate text with the trained model
   - Use restoration arguments for checkpoint loading

## Getting Started

### Prerequisites

```bash
pip install jax flax optax grain orbax tiktoken matplotlib
```

### Running the Notebooks

1. Start with `Building the LLM Architecture.ipynb` to understand the model structure
2. Run `Data Loading with Grain.ipynb` to prepare datasets
3. Execute `Training the Model.ipynb` to train the model
4. Use `Loading and Running a Pre-trained LLM.ipynb` for inference

## Model Configuration

Key parameters can be adjusted in the training notebook:

- `maxlen`: Sequence length (default: 128)
- `embed_dim`: Embedding dimension
- `num_heads`: Number of attention heads
- `ff_dim`: Feed-forward dimension
- `batch_size`: Training batch size (default: 32)
- `num_epochs`: Number of training epochs (default: 3)

## Dataset

The project uses TinyStories dataset (`TinyStories-1000.txt`), a collection of short stories separated by `<|endoftext|>` tokens. Stories are tokenized using the GPT-2 tokenizer which has a vocabulary size of 50,257.

## Features

- Efficient JAX/Flax implementation
- Causal self-attention for autoregressive generation
- Warmup + cosine decay learning rate schedule
- Checkpoint management with Orbax
- Device sharding for multi-device training
- Text generation from trained models
