"""
Tests for FractalCortex - Vic-Torch and FractalStack integration
"""

import torch
from fractal_cortex import (
    VicTorchAttention,
    FractalBlock,
    FractalStack,
    FractalCortex,
    create_fractal_cortex,
)


def test_vic_torch_attention():
    """Test VicTorchAttention component."""
    print("Testing VicTorchAttention...")
    
    dim = 256
    num_heads = 8
    batch_size = 2
    seq_len = 16
    
    attn = VicTorchAttention(dim=dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, dim)
    
    output, attn_weights = attn(x)
    
    assert output.shape == (batch_size, seq_len, dim), "Output shape mismatch"
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), "Attention weights shape mismatch"
    
    # Test self-awareness features
    stats = attn.get_attention_stats()
    assert "entropy" in stats, "Missing entropy in stats"
    assert "count" in stats, "Missing count in stats"
    
    print("✓ VicTorchAttention tests passed")


def test_fractal_block():
    """Test FractalBlock component."""
    print("Testing FractalBlock...")
    
    dim = 256
    num_heads = 8
    batch_size = 2
    seq_len = 16
    depth = 2
    
    block = FractalBlock(dim=dim, num_heads=num_heads, depth=depth)
    x = torch.randn(batch_size, seq_len, dim)
    
    output, attention_maps = block(x)
    
    assert output.shape == (batch_size, seq_len, dim), "Output shape mismatch"
    assert len(attention_maps) == depth, f"Expected {depth} attention maps, got {len(attention_maps)}"
    
    # Test depth stats
    stats = block.get_depth_stats()
    assert stats["depth"] == depth, "Depth mismatch in stats"
    
    print("✓ FractalBlock tests passed")


def test_fractal_stack():
    """Test FractalStack component."""
    print("Testing FractalStack...")
    
    dim = 256
    num_blocks = 4
    num_heads = 8
    fractal_depth = 2
    batch_size = 2
    seq_len = 16
    
    stack = FractalStack(
        dim=dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        fractal_depth=fractal_depth,
    )
    x = torch.randn(batch_size, seq_len, dim)
    
    output, all_attention_maps = stack(x)
    
    assert output.shape == (batch_size, seq_len, dim), "Output shape mismatch"
    assert len(all_attention_maps) == num_blocks, f"Expected {num_blocks} blocks, got {len(all_attention_maps)}"
    
    # Test stack stats
    stats = stack.get_stack_stats()
    assert len(stats) == num_blocks, "Stack stats count mismatch"
    
    print("✓ FractalStack tests passed")


def test_fractal_cortex_basic():
    """Test basic FractalCortex functionality."""
    print("Testing FractalCortex (basic)...")
    
    model = FractalCortex(
        input_dim=256,
        num_blocks=4,
        num_heads=8,
        fractal_depth=2,
        num_classes=10,
    )
    
    batch_size = 2
    seq_len = 16
    dim = 256
    
    x = torch.randn(batch_size, seq_len, dim)
    output = model(x)
    
    assert output.shape == (batch_size, 10), "Output shape mismatch for classification"
    
    print("✓ FractalCortex basic tests passed")


def test_fractal_cortex_feature_extraction():
    """Test FractalCortex in feature extraction mode."""
    print("Testing FractalCortex (feature extraction)...")
    
    model = FractalCortex(
        input_dim=256,
        num_blocks=4,
        num_heads=8,
        fractal_depth=2,
        num_classes=None,  # No classification head
    )
    
    batch_size = 2
    seq_len = 16
    dim = 256
    
    x = torch.randn(batch_size, seq_len, dim)
    output = model(x)
    
    assert output.shape == (batch_size, seq_len, dim), "Output shape mismatch for feature extraction"
    
    print("✓ FractalCortex feature extraction tests passed")


def test_fractal_cortex_attention_maps():
    """Test attention map extraction."""
    print("Testing attention map extraction...")
    
    model = create_fractal_cortex(model_size="tiny", num_classes=10)
    
    batch_size = 2
    seq_len = 16
    dim = 192
    
    x = torch.randn(batch_size, seq_len, dim)
    output, attention_maps = model(x, return_attention=True)
    
    assert output.shape == (batch_size, 10), "Output shape mismatch"
    assert len(attention_maps) > 0, "No attention maps returned"
    
    print("✓ Attention map extraction tests passed")


def test_fractal_cortex_image_input():
    """Test FractalCortex with image-like 4D input."""
    print("Testing FractalCortex (image input)...")
    
    model = FractalCortex(
        input_dim=256,
        num_blocks=4,
        num_heads=8,
        fractal_depth=2,
        num_classes=100,
    )
    
    batch_size = 2
    channels = 256
    height, width = 7, 7
    
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)
    
    assert output.shape == (batch_size, 100), "Output shape mismatch for image input"
    
    print("✓ FractalCortex image input tests passed")


def test_introspection():
    """Test self-awareness introspection features."""
    print("Testing introspection...")
    
    model = create_fractal_cortex(model_size="tiny", num_classes=10)
    
    # Run a few forward passes
    x = torch.randn(2, 10, 192)
    for _ in range(3):
        _ = model(x)
    
    # Introspect
    stats = model.introspect()
    
    assert stats["forward_count"] == 3, "Forward count mismatch"
    assert "parameter_count" in stats, "Missing parameter count"
    assert "trainable_parameters" in stats, "Missing trainable parameters"
    assert "stack_stats" in stats, "Missing stack stats"
    
    print("✓ Introspection tests passed")


def test_factory_function():
    """Test create_fractal_cortex factory function."""
    print("Testing factory function...")
    
    sizes = ["tiny", "small", "base", "large"]
    
    for size in sizes:
        model = create_fractal_cortex(model_size=size, num_classes=10)
        assert model is not None, f"Failed to create {size} model"
        assert isinstance(model, FractalCortex), f"{size} model is not FractalCortex instance"
    
    # Test invalid size
    try:
        create_fractal_cortex(model_size="invalid")
        assert False, "Should have raised ValueError for invalid size"
    except ValueError:
        pass  # Expected
    
    print("✓ Factory function tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Running FractalCortex Tests")
    print("=" * 80)
    print()
    
    tests = [
        test_vic_torch_attention,
        test_fractal_block,
        test_fractal_stack,
        test_fractal_cortex_basic,
        test_fractal_cortex_feature_extraction,
        test_fractal_cortex_attention_maps,
        test_fractal_cortex_image_input,
        test_introspection,
        test_factory_function,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {type(e).__name__}: {e}")
            failed += 1
            raise  # Re-raise to preserve stack trace
        print()
    
    print("=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
