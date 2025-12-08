"""
Test to verify that xFormers operations work correctly with fp16 dtype casting.
This test ensures that tensors are properly cast to fp16 before xFormers ops
and converted back to the original dtype after.
"""

import torch
import pytest

# Check if xFormers is available
try:
    from xformers.ops import scaled_index_add, index_select_cat, memory_efficient_attention
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    pytest.skip("xFormers not available", allow_module_level=True)

from dinov2.layers.block import get_attn_bias_and_cat, add_residual, get_branges_scales
from dinov2.layers.attention import MemEffAttention


class TestXFormersDtype:
    """Test xFormers dtype handling"""

    def test_scaled_index_add_fp32(self):
        """Test scaled_index_add with fp32 input"""
        torch.manual_seed(42)
        b, n, d = 4, 10, 64
        x = torch.randn(b, n, d, dtype=torch.float32)
        brange = torch.tensor([0, 1], dtype=torch.long)
        residual = torch.randn(2, n, d, dtype=torch.float32)
        scaling_vector = torch.ones(d, dtype=torch.float32)
        
        # This should not raise an error
        result = add_residual(x, brange, residual, 2.0, scaling_vector=scaling_vector)
        
        # Result should be in the original dtype
        assert result.dtype == torch.float32
        assert result.shape == x.shape

    def test_scaled_index_add_bfloat16(self):
        """Test scaled_index_add with bfloat16 input"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for bfloat16 test")
            
        torch.manual_seed(42)
        b, n, d = 4, 10, 64
        x = torch.randn(b, n, d, dtype=torch.bfloat16, device='cuda')
        brange = torch.tensor([0, 1], dtype=torch.long, device='cuda')
        residual = torch.randn(2, n, d, dtype=torch.bfloat16, device='cuda')
        scaling_vector = torch.ones(d, dtype=torch.bfloat16, device='cuda')
        
        # This should not raise an error
        result = add_residual(x, brange, residual, 2.0, scaling_vector=scaling_vector)
        
        # Result should be in the original dtype
        assert result.dtype == torch.bfloat16
        assert result.shape == x.shape

    def test_index_select_cat_fp32(self):
        """Test index_select_cat with fp32 input"""
        torch.manual_seed(42)
        x_list = [
            torch.randn(2, 10, 64, dtype=torch.float32),
            torch.randn(3, 15, 64, dtype=torch.float32),
        ]
        branges = [
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 1, 2], dtype=torch.long),
        ]
        
        # This should not raise an error
        attn_bias, cat_tensors = get_attn_bias_and_cat(x_list, branges)
        
        # Result should be in the original dtype
        assert cat_tensors.dtype == torch.float32

    def test_index_select_cat_bfloat16(self):
        """Test index_select_cat with bfloat16 input"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for bfloat16 test")
            
        torch.manual_seed(42)
        x_list = [
            torch.randn(2, 10, 64, dtype=torch.bfloat16, device='cuda'),
            torch.randn(3, 15, 64, dtype=torch.bfloat16, device='cuda'),
        ]
        branges = [
            torch.tensor([0, 1], dtype=torch.long, device='cuda'),
            torch.tensor([0, 1, 2], dtype=torch.long, device='cuda'),
        ]
        
        # This should not raise an error
        attn_bias, cat_tensors = get_attn_bias_and_cat(x_list, branges)
        
        # Result should be in the original dtype
        assert cat_tensors.dtype == torch.bfloat16

    def test_memory_efficient_attention_fp32(self):
        """Test MemEffAttention with fp32 input"""
        torch.manual_seed(42)
        dim = 64
        num_heads = 8
        attn = MemEffAttention(dim=dim, num_heads=num_heads)
        
        x = torch.randn(2, 10, dim, dtype=torch.float32)
        
        # This should not raise an error
        output = attn(x)
        
        # Output should be in the original dtype
        assert output.dtype == torch.float32
        assert output.shape == x.shape

    def test_memory_efficient_attention_bfloat16(self):
        """Test MemEffAttention with bfloat16 input"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for bfloat16 test")
            
        torch.manual_seed(42)
        dim = 64
        num_heads = 8
        attn = MemEffAttention(dim=dim, num_heads=num_heads).cuda()
        
        x = torch.randn(2, 10, dim, dtype=torch.bfloat16, device='cuda')
        
        # This should not raise an error
        output = attn(x)
        
        # Output should be in the original dtype
        assert output.dtype == torch.bfloat16
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
