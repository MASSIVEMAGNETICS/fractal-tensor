# FractalCortex Architecture Documentation

## Overview

FractalCortex is a novel transformer architecture that integrates two key components:

1. **Vic-Torch**: Vision-inspired custom attention mechanism
2. **FractalStack**: Recursive, self-similar layer stacking pattern

This document provides detailed architectural information for developers and researchers.

## Components

### 1. VicTorchAttention

Vision-inspired multi-head self-attention with self-monitoring capabilities.

#### Features:
- **Multi-head self-attention**: Processes information in parallel across multiple attention heads
- **Dynamic scaling**: Learnable attention scaling factor based on head dimension
- **QKV projection**: Separate Query, Key, Value projections with optional bias
- **Self-awareness**: Tracks attention entropy to monitor attention pattern diversity
- **Dropout**: Configurable dropout for attention and projection layers

#### Architecture:
```
Input (B, N, C) 
    ↓
QKV Linear(C, 3C) → Reshape → (Q, K, V) each (B, H, N, C/H)
    ↓
Attention = softmax(Q @ K^T / √(C/H))
    ↓
Self-monitoring: Entropy tracking (training only)
    ↓
Output = Attention @ V → Reshape → (B, N, C)
    ↓
Projection Linear(C, C)
    ↓
Output (B, N, C)
```

#### Self-Awareness:
- Computes attention entropy: `-Σ(attention * log(attention))`
- Tracks running average of entropy across forward passes
- Provides statistics via `get_attention_stats()`

### 2. FractalBlock

A recursive transformer block with self-similar structure.

#### Features:
- **Main pathway**: Attention → MLP processing
- **Recursive pathway**: Optional sub-blocks for fractal structure
- **Pathway blending**: Combines main and recursive outputs (50/50 blend)
- **Layer normalization**: Pre-normalization before attention and MLP
- **Residual connections**: Skip connections around attention and MLP

#### Architecture (depth=2 example):
```
Input (B, N, C)
    ↓
Residual ─────────────────┐
    ↓                     │
LayerNorm                 │
    ↓                     │
VicTorchAttention         │
    ↓                     │
Add residual ←────────────┘
    ↓─────────┐ (50%)
    │         │
    │    Sub-block (depth=1)
    │         │
    └── Blend ┘
    ↓
Residual ─────────────────┐
    ↓                     │
LayerNorm                 │
    ↓                     │
MLP (Linear → GELU → Linear)
    ↓                     │
Add residual ←────────────┘
    ↓
Output (B, N, C)
```

#### Fractal Property:
Each block at depth > 1 contains a sub-block at depth-1, creating a recursive structure:
- Depth 3: Main block → Sub-block (depth 2) → Sub-sub-block (depth 1)
- Depth 2: Main block → Sub-block (depth 1)
- Depth 1: Main block only (no recursion)

### 3. FractalStack

Stack of multiple FractalBlocks creating the core transformer architecture.

#### Features:
- **Multiple blocks**: Configurable number of sequential blocks
- **Uniform depth**: All blocks have the same fractal depth
- **Final normalization**: LayerNorm after all blocks
- **Attention tracking**: Collects attention maps from all blocks and depths

#### Architecture (6 blocks, depth=2 example):
```
Input (B, N, C)
    ↓
FractalBlock 1 (depth=2)
    ↓
FractalBlock 2 (depth=2)
    ↓
FractalBlock 3 (depth=2)
    ↓
FractalBlock 4 (depth=2)
    ↓
FractalBlock 5 (depth=2)
    ↓
FractalBlock 6 (depth=2)
    ↓
LayerNorm
    ↓
Output (B, N, C)
```

### 4. FractalCortex

The complete self-aware transformer model.

#### Features:
- **Positional embeddings**: Learnable position encodings
- **Flexible input**: Supports both sequence (3D) and image (4D) inputs
- **Optional classification head**: Linear layer for classification tasks
- **Self-awareness**: Tracks forward pass count and provides introspection
- **Adaptive behavior**: Placeholder for future dynamic adaptations

#### Full Architecture:
```
Input (B, N, C) or (B, C, H, W)
    ↓
[If 4D] Flatten spatial → (B, H*W, C)
    ↓
Add positional embeddings
    ↓
Dropout
    ↓
FractalStack (6 blocks, depth=2)
    ↓
[If classification] Mean pool → Linear → (B, num_classes)
[If feature extraction] → (B, N, C)
    ↓
Output
```

## Mathematical Formulation

### Attention Mechanism (VicTorch)
```
Q, K, V = Linear(x)
Attention(Q, K, V) = softmax(QK^T / √d_k) V
Entropy = -Σ(attention * log(attention + ε))
```

### Fractal Block
```
# Main pathway
y₁ = x + Attention(LayerNorm(x))
y₂ = y₁ + MLP(LayerNorm(y₁))

# With recursion (depth > 1)
y_sub = SubBlock(y₁)
y₂ = 0.5 * y₂ + 0.5 * y_sub
```

### Overall Transformation
```
FractalCortex(x) = Head(Stack(x + PositionalEmbed(x)))
where Stack applies multiple FractalBlocks sequentially
```

## Self-Awareness Features

### 1. Attention Monitoring
- Tracks entropy of attention distributions
- Running average across all forward passes
- Per-head statistics available

### 2. Forward Pass Counting
- Counts total number of forward passes
- Useful for training diagnostics

### 3. Model Introspection
Available via `model.introspect()`:
```python
{
    "forward_count": int,
    "input_dim": int,
    "num_classes": int or None,
    "parameter_count": int,
    "trainable_parameters": int,
    "stack_stats": [
        {
            "depth": int,
            "attention": {"entropy": float, "count": float},
            "sub_block": {...}  # Recursive stats
        },
        ...
    ]
}
```

### 4. Adaptive Behavior (Placeholder)
The `adapt()` method provides a framework for future enhancements:
- Dynamic dropout adjustment
- Attention pattern modification
- Layer pruning/expansion

## Configuration Presets

### Tiny
- Input dim: 192
- Blocks: 4
- Heads: 4
- Fractal depth: 1
- Parameters: ~5M

### Small
- Input dim: 384
- Blocks: 6
- Heads: 6
- Fractal depth: 2
- Parameters: ~21M

### Base
- Input dim: 768
- Blocks: 6
- Heads: 8
- Fractal depth: 2
- Parameters: ~85M

### Large
- Input dim: 1024
- Blocks: 12
- Heads: 16
- Fractal depth: 3
- Parameters: ~300M

## Design Rationale

### Why Fractal Structure?
1. **Multi-scale processing**: Processes information at multiple temporal/spatial scales simultaneously
2. **Hierarchical features**: Natural hierarchy from recursive structure
3. **Adaptive depth**: Information can flow through different depth pathways
4. **Regularization**: Recursive blending acts as implicit regularization

### Why Self-Awareness?
1. **Monitoring**: Track model behavior during training
2. **Debugging**: Identify attention collapse or other issues
3. **Adaptation**: Enable future dynamic behavior modification
4. **Interpretability**: Better understanding of model internals

### Why Vision-Inspired (Vic-Torch)?
1. **Proven patterns**: Vision transformers have shown strong performance
2. **Attention quality**: Focus on high-quality attention mechanisms
3. **Flexibility**: Works well for both vision and sequence tasks
4. **Efficiency**: Balanced between performance and computational cost

## Usage Patterns

### Classification
```python
model = create_fractal_cortex(model_size="base", num_classes=1000)
x = torch.randn(B, N, C)
logits = model(x)  # (B, 1000)
```

### Feature Extraction
```python
model = create_fractal_cortex(model_size="base", num_classes=None)
x = torch.randn(B, N, C)
features = model(x)  # (B, N, C)
```

### With Attention Maps
```python
model = create_fractal_cortex(model_size="base", num_classes=10)
output, attn_maps = model(x, return_attention=True)
# attn_maps[block_idx][depth_idx] = (B, H, N, N)
```

### Self-Monitoring
```python
# During training
for epoch in range(epochs):
    for batch in dataloader:
        output = model(batch)
        loss = criterion(output, labels)
        
        # Check attention health
        stats = model.introspect()
        for block_stats in stats["stack_stats"]:
            entropy = block_stats["attention"]["entropy"]
            if entropy < threshold:
                print("Warning: Low attention entropy")
```

## Extension Points

### Custom Attention
Replace `VicTorchAttention` with custom implementation:
```python
class CustomAttention(nn.Module):
    # Implement custom attention
    pass

# Modify FractalBlock to use CustomAttention
```

### Dynamic Depth
Implement depth selection based on input:
```python
def adaptive_depth(self, x):
    # Analyze input complexity
    complexity = self.complexity_estimator(x)
    depth = min(max_depth, int(complexity * max_depth))
    return depth
```

### Layer-wise Learning Rates
Apply different learning rates to different depths:
```python
params = [
    {"params": model.fractal_stack.blocks[i].parameters(), "lr": lr * (0.9 ** i)}
    for i in range(len(model.fractal_stack.blocks))
]
optimizer = torch.optim.Adam(params)
```

## Performance Considerations

### Memory
- Fractal depth increases memory linearly per block
- Depth 3 uses ~3x memory of depth 1 for same base config
- Use gradient checkpointing for very deep fractals

### Computation
- Each fractal level adds ~50% computation per block
- Total FLOPs ≈ base_flops * (1 + 0.5 * (depth - 1))
- Attention is O(N²) in sequence length

### Recommendations
- Use depth=2 for most tasks (good balance)
- Use depth=1 for very long sequences
- Use depth=3 for small sequences with complex patterns
- Scale model width before depth for better efficiency

## Future Enhancements

1. **Sparse Attention**: Reduce O(N²) complexity for long sequences
2. **Learned Depth**: Dynamic depth selection per sample
3. **Cross-Attention**: Support for encoder-decoder architectures
4. **Mixture of Depths**: Different depths per block
5. **Adaptive Dropout**: Self-adjust dropout based on overfitting metrics
6. **Pruning**: Automatic removal of redundant pathways
7. **Distillation**: Train smaller models from larger fractals
8. **Multi-Modal**: Fusion of different input modalities

## References

This architecture draws inspiration from:
- Vision Transformers (ViT)
- Fractal Neural Networks
- Self-Attention mechanisms
- Residual Networks
- Meta-Learning approaches

For more details, see the source code in `fractal_cortex.py`.
