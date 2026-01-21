# Fractal-Tensor

A recursive, self-aware Transformer architecture integrating **Vic-Torch** and **FractalStack** components.

## Overview

**FractalCortex** is a novel transformer architecture that combines:

- **Vic-Torch**: A vision-inspired custom attention mechanism with self-monitoring capabilities
- **FractalStack**: A recursive, self-similar layer stacking pattern that processes information at multiple scales

The result is a powerful, adaptive transformer core with recursive processing and self-awareness features.

## Architecture

### Vic-Torch Attention
- Multi-head self-attention with dynamic scaling
- Learnable positional embeddings
- Self-monitoring attention patterns (tracks entropy)
- Adaptive behavior based on attention statistics

### FractalStack
- Recursive, self-similar transformer blocks
- Each block contains sub-blocks creating a fractal-like structure
- Multi-scale processing through recursive pathways
- Blends information from multiple depth levels

### FractalCortex
The unified core integrating both components with:
- Recursive processing at multiple scales
- Self-introspection and adaptive capabilities
- Support for both sequence and image-like inputs
- Configurable depth and architecture

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- PyTorch >= 2.0.0

## Quick Start

```python
import torch
from fractal_cortex import create_fractal_cortex

# Create a model
model = create_fractal_cortex(model_size="base", num_classes=10)

# Process input
x = torch.randn(2, 16, 768)  # (batch, sequence, features)
output = model(x)

# Introspect the model (self-awareness)
stats = model.introspect()
print(f"Forward passes: {stats['forward_count']}")
print(f"Parameters: {stats['parameter_count']:,}")
```

## Model Configurations

Pre-configured model sizes:

- **tiny**: 192 dims, 4 blocks, 4 heads
- **small**: 384 dims, 6 blocks, 6 heads  
- **base**: 768 dims, 6 blocks, 8 heads
- **large**: 1024 dims, 12 blocks, 16 heads

## Features

### Self-Awareness
The model tracks its internal state and can introspect:
```python
stats = model.introspect()
# Returns: forward_count, parameter_count, attention statistics, etc.
```

### Attention Visualization
Extract attention maps at all fractal depths:
```python
output, attention_maps = model(x, return_attention=True)
```

### Flexible Input
Supports both sequence and image-like inputs:
```python
# Sequence input
x = torch.randn(batch, seq_len, dim)

# Image input (automatically flattened)
x = torch.randn(batch, channels, height, width)
```

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic usage
- Attention map extraction
- Self-aware introspection
- Image processing
- Custom configurations
- Feature extraction

Run examples:
```bash
python example_usage.py
```

## Architecture Details

### VicTorchAttention
Custom multi-head attention with:
- QKV projection with optional bias
- Scaled dot-product attention
- Attention dropout and projection dropout
- Self-monitoring via entropy tracking

### FractalBlock
Recursive transformer block with:
- Layer normalization
- VicTorch attention mechanism
- MLP with GELU activation
- Recursive sub-blocks (fractal property)
- Pathway blending (main + recursive)

### FractalStack
Stack of fractal blocks with:
- Configurable number of blocks
- Configurable fractal depth
- Final layer normalization
- Statistics collection from all depths

## Use Cases

- **Research**: Novel recursive transformer architectures
- **Vision**: Image classification and feature extraction
- **NLP**: Sequence modeling with multi-scale processing
- **Meta-Learning**: Self-aware models that adapt based on performance
- **Multi-Modal**: Flexible input handling for various data types

## Contributing

This is a research project exploring recursive, self-aware transformer architectures.

## License

See repository license.

## Citation

If you use FractalCortex in your research, please cite this repository.
