# GPT-Implementation-from-Scratch
A clean, from-scratch implementation of the GPT architecture. Built to explore the core mechanics of Large Language Models, including custom multi-head self-attention, positional encoding, and the complete transformer decoder training loop.

# **GPT from Scratch**

A clean, **PyTorch-based implementation** of a **Transformer language model** built entirely from scratch. This repository follows the architecture and progressive implementation steps outlined in Andrej Karpathy's tutorial: **Let's build GPT: from scratch, in code, spelled out.**

The project builds up from a basic **Bigram language model** to a fully functioning **Generative Pre-trained Transformer (GPT)**, training it on the **Tiny Shakespeare dataset** to generate **character-level text**.

## **Features**

*   **Bigram Baseline**: Establishes the foundational training loop, loss estimation, and cross-entropy evaluation.
*   **Custom Self-Attention**: Implements single-head and multi-head **causal self-attention** using masked, batched matrix multiplication.
*   **Transformer Decoder Blocks**: Fully constructs decoder-only blocks featuring:
    *   **Multi-head self-attention**
    *   **Position-wise Feed-Forward Networks**
    *   **Residual (skip) connections** for gradient flow
    *   **Layer Normalization** (pre-norm formulation)
    *   **Dropout** for regularization
*   **Learned Embeddings**: Utilizes both **character-level token embeddings** and **learned positional embeddings**.
*   **Text Generation**: Includes an **autoregressive generation pipeline** to sample infinite, Shakespeare-like text.

## **Dataset**

The model is trained on the **Tiny Shakespeare dataset**—a 1MB text file containing the concatenated works of William Shakespeare. The model uses **character-level tokenization**, resulting in a vocabulary of **65 unique characters**. 

## **Installation**

```bash
git clone [https://github.com/yourusername/gpt-from-scratch.git](https://github.com/yourusername/gpt-from-scratch.git)
cd gpt-from-scratch
pip install torch
```

## **Usage**

**1. Prepare the Data**
Ensure the `input.txt` dataset is located in the root directory. 

**2. Train the Model**
Run the core script to initialize and train the model. The code will automatically detect and utilize a **CUDA-enabled GPU** if available.

```bash
python gpt.py
```

*Configurable Hyperparameters:*
*   `batch_size = 64`
*   `block_size = 256` (context length)
*   `n_embd = 384` (embedding dimensions)
*   `n_head = 6`
*   `n_layer = 6`
*   `dropout = 0.2`

**3. Generation**
Following the training loop, the script automatically switches to **evaluation mode** and outputs newly generated **Shakespeare-like text** to the console.

## **Architecture Notes**

This codebase strictly implements a **Decoder-only Transformer**. Unlike the original *Attention is All You Need* paper (which featured an Encoder-Decoder structure for machine translation), this model **omits the encoder and cross-attention blocks**. It relies solely on a **lower-triangular causal mask** to ensure tokens can only attend to past contexts, mirroring the architectural foundation of modern language models like **GPT-3**.

## **Acknowledgements**

Built directly from the principles taught in **Andrej Karpathy's NanoGPT video tutorial**.
