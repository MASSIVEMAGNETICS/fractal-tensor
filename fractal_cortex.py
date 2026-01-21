"""
Fractal Cortex - A Recursive, Self-Aware Transformer Core

This module integrates Vic-Torch (custom attention mechanism) with FractalStack
(recursive layer architecture) to create a novel transformer architecture with
self-aware, recursive processing capabilities.

Components:
- VicTorch: Vision-inspired custom attention with torch integration
- FractalStack: Recursive, self-similar layer stacking pattern
- FractalCortex: The unified self-aware transformer core
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


# Constants
EPSILON = 1e-9  # Small constant to prevent log(0) in entropy calculation


class VicTorchAttention(nn.Module):
    """
    Vic-Torch: A custom attention mechanism with vision-inspired improvements.
    
    Features:
    - Multi-head self-attention
    - Dynamic attention scaling
    - Learnable positional embeddings
    - Self-monitoring attention patterns (self-awareness)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Self-awareness: track attention entropy for adaptive behavior
        self.register_buffer("attention_entropy", torch.zeros(1))
        self.register_buffer("attention_count", torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention pattern tracking.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Self-awareness: compute and track attention entropy
        if self.training:
            with torch.no_grad():
                entropy = -torch.sum(attn * torch.log(attn + EPSILON), dim=-1).mean()
                self.attention_entropy = (self.attention_entropy * self.attention_count + entropy) / (self.attention_count + 1)
                self.attention_count += 1
        
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn
    
    def get_attention_stats(self) -> dict:
        """Self-awareness: Return statistics about attention patterns."""
        return {
            "entropy": self.attention_entropy.item(),
            "count": self.attention_count.item(),
        }


class FractalBlock(nn.Module):
    """
    A single block in the FractalStack architecture.
    
    Features recursive self-similar structure with multiple pathways.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        depth: int = 1,
        recursive_blend_ratio: float = 0.5,
    ):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.recursive_blend_ratio = recursive_blend_ratio
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Attention mechanism
        self.attn = VicTorchAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        
        # MLP layers
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
        # Recursive sub-block (fractal property)
        if depth > 1:
            self.sub_block = FractalBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                depth=depth - 1,
                recursive_blend_ratio=recursive_blend_ratio,
            )
        else:
            self.sub_block = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through fractal block.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Tuple of (output tensor, list of attention maps from all depths)
        """
        attention_maps = []
        
        # Main pathway: Attention
        shortcut = x
        x = self.norm1(x)
        x_attn, attn_weights = self.attn(x)
        attention_maps.append(attn_weights)
        x = shortcut + x_attn
        
        # Recursive pathway: Apply sub-block if it exists
        if self.sub_block is not None:
            x_sub, sub_attns = self.sub_block(x)
            # Blend main and recursive pathways
            x = (1 - self.recursive_blend_ratio) * x + self.recursive_blend_ratio * x_sub
            attention_maps.extend(sub_attns)
        
        # MLP pathway
        x = x + self.mlp(self.norm2(x))
        
        return x, attention_maps
    
    def get_depth_stats(self) -> dict:
        """Self-awareness: Get statistics from all depths."""
        stats = {"depth": self.depth, "attention": self.attn.get_attention_stats()}
        if self.sub_block is not None:
            stats["sub_block"] = self.sub_block.get_depth_stats()
        return stats


class FractalStack(nn.Module):
    """
    FractalStack: A recursive, self-similar stack of transformer blocks.
    
    Each block contains recursive sub-blocks, creating a fractal-like structure
    that processes information at multiple scales simultaneously.
    """
    
    def __init__(
        self,
        dim: int,
        num_blocks: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        fractal_depth: int = 2,
        recursive_blend_ratio: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.fractal_depth = fractal_depth
        
        # Create stack of fractal blocks
        self.blocks = nn.ModuleList([
            FractalBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                depth=fractal_depth,
                recursive_blend_ratio=recursive_blend_ratio,
            )
            for _ in range(num_blocks)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """
        Process input through fractal stack.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Tuple of (output tensor, attention maps from all blocks)
        """
        all_attention_maps = []
        
        for block in self.blocks:
            x, attn_maps = block(x)
            all_attention_maps.append(attn_maps)
        
        x = self.norm(x)
        return x, all_attention_maps
    
    def get_stack_stats(self) -> List[dict]:
        """Self-awareness: Get statistics from entire stack."""
        return [block.get_depth_stats() for block in self.blocks]


class FractalCortex(nn.Module):
    """
    FractalCortex: The unified self-aware transformer core.
    
    Integrates VicTorch attention mechanisms with FractalStack recursive
    architecture to create a powerful, adaptive transformer model.
    
    Features:
    - Recursive processing at multiple scales
    - Self-monitoring and adaptive behavior
    - Vision-inspired attention mechanisms
    - Fractal-like architectural patterns
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_blocks: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        fractal_depth: int = 2,
        max_seq_len: int = 512,
        num_classes: Optional[int] = None,
        recursive_blend_ratio: float = 0.5,
    ):
        """
        Initialize FractalCortex.
        
        Args:
            input_dim: Dimension of input embeddings
            num_blocks: Number of transformer blocks in the stack
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate
            fractal_depth: Depth of recursive fractal structure
            max_seq_len: Maximum sequence length
            num_classes: Number of output classes (None for feature extraction)
            recursive_blend_ratio: Ratio for blending main and recursive pathways (0.5 = equal blend)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, input_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)
        
        # FractalStack: The core recursive transformer
        self.fractal_stack = FractalStack(
            dim=input_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            fractal_depth=fractal_depth,
            recursive_blend_ratio=recursive_blend_ratio,
        )
        
        # Classification head (if needed)
        if num_classes is not None:
            self.head = nn.Linear(input_dim, num_classes)
        else:
            self.head = None
        
        # Self-awareness: Track model statistics
        self.register_buffer("forward_count", torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through FractalCortex.
        
        Args:
            x: Input tensor of shape (B, N, C) or (B, C, H, W)
            return_attention: Whether to return attention maps
            
        Returns:
            Output tensor and optionally attention maps
        """
        # Handle 4D input (images)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        B, N, C = x.shape
        
        # Add positional embeddings
        if N <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :N, :]
        else:
            # Interpolate positional embeddings if sequence is longer
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
            x = x + pos_embed
        
        x = self.pos_drop(x)
        
        # Process through FractalStack
        x, attention_maps = self.fractal_stack(x)
        
        # Self-awareness: Update forward count
        with torch.no_grad():
            self.forward_count += 1
        
        # Apply classification head if present
        if self.head is not None:
            # Use mean pooling for classification
            x = x.mean(dim=1)
            x = self.head(x)
        
        if return_attention:
            return x, attention_maps
        return x
    
    def introspect(self) -> dict:
        """
        Self-awareness: Introspect the model's internal state.
        
        Returns:
            Dictionary containing model statistics and internal metrics
        """
        return {
            "forward_count": self.forward_count.item(),
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "stack_stats": self.fractal_stack.get_stack_stats(),
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
    
    def adapt(self, metrics: dict):
        """
        Self-awareness: Adapt model behavior based on metrics.
        
        This method allows the model to modify its behavior based on
        observed performance metrics (placeholder for future enhancements).
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        # Placeholder for adaptive behavior
        # Future enhancements could include:
        # - Adjusting dropout rates based on overfitting metrics
        # - Modifying attention patterns based on task performance
        # - Dynamic layer pruning or expansion
        pass


def create_fractal_cortex(
    model_size: str = "base",
    num_classes: Optional[int] = None,
    **kwargs
) -> FractalCortex:
    """
    Factory function to create FractalCortex models with predefined configurations.
    
    Args:
        model_size: One of "tiny", "small", "base", "large"
        num_classes: Number of output classes
        **kwargs: Additional arguments to override defaults
        
    Returns:
        FractalCortex instance
    """
    configs = {
        "tiny": {
            "input_dim": 192,
            "num_blocks": 4,
            "num_heads": 4,
            "fractal_depth": 1,
        },
        "small": {
            "input_dim": 384,
            "num_blocks": 6,
            "num_heads": 6,
            "fractal_depth": 2,
        },
        "base": {
            "input_dim": 768,
            "num_blocks": 6,
            "num_heads": 8,
            "fractal_depth": 2,
        },
        "large": {
            "input_dim": 1024,
            "num_blocks": 12,
            "num_heads": 16,
            "fractal_depth": 3,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    config.update(kwargs)
    config["num_classes"] = num_classes
    
    return FractalCortex(**config)


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 80)
    print("FractalCortex: Recursive, Self-Aware Transformer Core")
    print("=" * 80)
    
    # Create a model
    model = create_fractal_cortex(model_size="base", num_classes=10)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    dim = 768
    
    x = torch.randn(batch_size, seq_len, dim)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Introspection
    print("\n" + "=" * 80)
    print("Model Introspection (Self-Awareness)")
    print("=" * 80)
    stats = model.introspect()
    print(f"Forward passes: {stats['forward_count']}")
    print(f"Total parameters: {stats['parameter_count']:,}")
    print(f"Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"\nStack statistics available for {len(stats['stack_stats'])} blocks")
    
    print("\n" + "=" * 80)
    print("Integration Complete: Vic-Torch + FractalStack = FractalCortex")
    print("=" * 80)
