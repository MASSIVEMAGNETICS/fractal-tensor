"""
Example usage of FractalCortex - demonstrating the integration of
Vic-Torch and FractalStack components.
"""

import torch
from fractal_cortex import FractalCortex, create_fractal_cortex


def example_basic_usage():
    """Basic usage example."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    
    # Create a model for classification
    model = create_fractal_cortex(model_size="small", num_classes=100)
    
    # Create sample input
    batch_size = 4
    seq_len = 32
    input_dim = 384
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    

def example_with_attention():
    """Example showing attention map extraction."""
    print("\n" + "=" * 80)
    print("Example 2: Extracting Attention Maps")
    print("=" * 80)
    
    model = create_fractal_cortex(model_size="tiny", num_classes=10)
    
    # Sample input
    x = torch.randn(2, 16, 192)
    
    # Forward pass with attention maps
    output, attention_maps = model(x, return_attention=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Number of blocks: {len(attention_maps)}")
    print(f"Attention maps per block (fractal depth): {len(attention_maps[0])}")
    print(f"First attention map shape: {attention_maps[0][0].shape}")


def example_introspection():
    """Example demonstrating self-awareness through introspection."""
    print("\n" + "=" * 80)
    print("Example 3: Self-Aware Introspection")
    print("=" * 80)
    
    model = create_fractal_cortex(model_size="base")
    
    # Run multiple forward passes
    x = torch.randn(2, 10, 768)
    for i in range(5):
        _ = model(x)
    
    # Introspect the model
    stats = model.introspect()
    
    print(f"Forward passes executed: {stats['forward_count']}")
    print(f"Input dimension: {stats['input_dim']}")
    print(f"Total parameters: {stats['parameter_count']:,}")
    print(f"Trainable parameters: {stats['trainable_parameters']:,}")
    
    # Show statistics from the first block
    if stats['stack_stats']:
        block_stats = stats['stack_stats'][0]
        print(f"\nFirst block depth: {block_stats['depth']}")
        print(f"Attention entropy: {block_stats['attention']['entropy']:.4f}")


def example_image_input():
    """Example with image-like input (4D tensor)."""
    print("\n" + "=" * 80)
    print("Example 4: Processing Image-like Input")
    print("=" * 80)
    
    model = create_fractal_cortex(model_size="small", num_classes=1000)
    
    # Simulate image patches (e.g., from a CNN backbone)
    batch_size = 2
    channels = 384
    height, width = 7, 7  # Spatial dimensions
    
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape (B, C, H, W): {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    

def example_custom_configuration():
    """Example with custom configuration."""
    print("\n" + "=" * 80)
    print("Example 5: Custom Configuration")
    print("=" * 80)
    
    # Create a custom model
    model = FractalCortex(
        input_dim=512,
        num_blocks=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.15,
        fractal_depth=3,  # Deeper fractal recursion
        max_seq_len=1024,
        num_classes=None,  # Feature extraction mode
    )
    
    x = torch.randn(2, 64, 512)
    features = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Fractal depth: 3 (deeper recursion)")
    
    # Introspect
    stats = model.introspect()
    print(f"Parameters: {stats['parameter_count']:,}")


def example_feature_extraction():
    """Example using the model for feature extraction."""
    print("\n" + "=" * 80)
    print("Example 6: Feature Extraction Mode")
    print("=" * 80)
    
    # Model without classification head
    model = create_fractal_cortex(model_size="base", num_classes=None)
    
    # Input sequence
    x = torch.randn(1, 50, 768)
    
    # Extract features
    features = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output features shape: {features.shape}")
    print("Features can be used for downstream tasks")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FRACTAL CORTEX - Example Usage")
    print("Vic-Torch + FractalStack Integration")
    print("=" * 80 + "\n")
    
    # Run all examples
    example_basic_usage()
    example_with_attention()
    example_introspection()
    example_image_input()
    example_custom_configuration()
    example_feature_extraction()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
