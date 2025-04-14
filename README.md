# iRoPE (Interleaved Rotary Position Embeddings)

This repository contains an experimental implementation of iRoPE (Interleaved Rotary Position Embeddings), a novel approach to enhancing transformers' long context handling capabilities by interleaving two complementary attention mechanisms:

1. **Local Attention with RoPE** (Rotary Position Embeddings): Handles local token relationships effectively
2. **Global Attention with Temperature Scaling**: Captures long-range dependencies without RoPE's periodicity issues

## ⚠️ Research Implementation Only

> **Note**: This implementation is intended for research and experimentation purposes only. For optimal results, models should be trained from scratch with this attention mechanism rather than retrofitting pretrained models.

## Overview

iRoPE addresses the challenge of handling extremely long contexts (10k-100k+ tokens) in transformer models through an interleaved architecture that combines:

- **Even-numbered layers**: Local attention with standard RoPE in fixed-size chunks
- **Odd-numbered layers**: Global attention with temperature-scaled queries (no RoPE)

This implementation creates a modified version of a pretrained LLaMA model by replacing its attention mechanism with the iRoPE approach, transferring weights from the original model.

## Key Features

- 🔄 Interleaves two attention mechanisms throughout the transformer stack
- 🧩 Processes extremely long contexts by breaking them into manageable chunks
- 🌡️ Implements various temperature scaling functions (log, linear, exponential, sigmoid, power-law)
- ⚡ Retrofits pre-trained LLaMA models to extend context length without retraining
- 📝 Simulates very long contexts (configurable up to millions of tokens)

## How It Works

1. **Local Attention Layers (Even)**: Apply traditional RoPE in fixed-size chunks (default: 2048 tokens) to maintain precise positional understanding within local windows
   
2. **Global Attention Layers (Odd)**: Apply temperature scaling to query vectors without RoPE to avoid periodicity issues at long distances
   
3. **Temperature Scaling Functions**:
   - Log scaling: `1 + log(floor(position / α) + 1) * β`
   - Linear scaling: `1 + (position / α) * β`
   - Exponential scaling: `1 + exp((position / α) - 1) * β`
   - Sigmoid scaling: `1 + sigmoid((position / α) - (L / (2*α))) * β`
   - Power-law scaling: `1 + (position / α)^γ * β`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/irope.git
cd irope
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Jupyter Notebook / JupyterLab (for UI interface)
- ipywidgets (for interactive UI)
- CUDA-compatible GPU with at least 16GB VRAM (recommended)

## Usage

### Quick Start

Open the Jupyter notebook `iRope.ipynb` and run all cells. Make sure to add your Hugging Face token in the config section.

### Configuration

The main hyperparameters can be adjusted in the "Hyperparams" section:

```python
# iRoPE Hyperparameters
CHUNK_SIZE = 2048          # Local attention chunk size
ALPHA = 8192               # α for temperature scaling
BETA = 0.1                 # β for temperature scaling
GAMMA = 0.5                # For power-law scaling
SCALING_TYPE = "log"       # "log", "linear", "exp", "sigmoid", "power"
MAX_SEQ_LEN = 16384        # Max processing chunk length (GPU memory dependent)
SIMULATED_CONTEXT_LENGTH = 100_000  # Context length to simulate
```

### Interactive UI

The notebook provides an interactive UI where you can:

1. Input text prompts
2. Configure simulated context length
3. Generate responses with the iRoPE-modified model

## Implementation Details

### Key Components

1. **`LocalAttentionWithRoPE`**: Implements chunked attention with standard RoPE
2. **`GlobalAttentionWithTempScaling`**: Implements global attention with temperature scaling
3. **`LlamaLayerWithIRoPE`**: A LLaMA decoder layer modified to use either local or global attention
4. **`LlamaWithIRoPE`**: Main model class that replaces standard layers with iRoPE layers
5. **`IRoPELlamaInterface`**: UI interface for interactive testing

### Weight Transfer Process

The implementation transfers weights from the original pretrained model to the iRoPE model:

1. Embedding weights
2. Layer normalization weights
3. Attention projection weights (Q, K, V, O)
4. MLP weights
5. Final layer norm weights
6. Language model head weights

## Performance Considerations

- Memory usage scales with chunk size - adjust `MAX_SEQ_LEN` based on your GPU's VRAM
- Larger context simulation will be slower - start with smaller values and increase as needed
- The implementation simulates long contexts by repeating the input tokens - this is a demonstration approach

## Experimental Results

The effectiveness of iRoPE depends significantly on the hyperparameters used, particularly:

- **α (alpha)**: Controls the rate of scaling growth
- **β (beta)**: Controls the magnitude of scaling
- **Chunk size**: Affects local context window size
- **Scaling function**: Different functions may work better for different models

## Limitations

- This implementation retrofits pretrained models rather than training from scratch
- Optimal performance would require training models with iRoPE from the beginning
- The weight transfer approach maintains the model's trained weights but changes the position encoding mechanism, which may affect performance
- Simulated long contexts are not the same as real-world long contexts
