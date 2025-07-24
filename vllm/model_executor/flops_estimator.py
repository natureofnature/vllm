"""
Model Configuration P-Value Estimator
estimate non attention and attention calculation ratio
"""

import json
import argparse
from typing import Dict, Tuple, List
import numpy as np


class ModelPEstimator:
    """estimate non attention to attention ratio"""
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        初始化估算器
        
        Args:
            config_path: path to config file
            config_dict: config dict
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_path or config_dict")
        
        self._parse_config()
    
    def _parse_config(self):
        """parse model config"""
        self.model_type = self.config.get('model_type', 'unknown')
        self.hidden_size = self.config.get('hidden_size')
        self.intermediate_size = self.config.get('intermediate_size')
        self.num_attention_heads = self.config.get('num_attention_heads')
        self.num_key_value_heads = self.config.get('num_key_value_heads', self.num_attention_heads)
        self.num_hidden_layers = self.config.get('num_hidden_layers')
        self.vocab_size = self.config.get('vocab_size', 32000)
        
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.gqa_ratio = self.num_attention_heads // self.num_key_value_heads
        self.ffn_expansion = self.intermediate_size / self.hidden_size
        
        if not all([self.hidden_size, self.intermediate_size, self.num_attention_heads]):
            raise ValueError("Missing required model parameters")
    
    def print_model_info(self):
        """print model info"""
        print(f"Model Type: {self.model_type}")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Intermediate Size: {self.intermediate_size} ({self.ffn_expansion:.1f}x expansion)")
        print(f"Attention Heads: {self.num_attention_heads}")
        print(f"KV Heads: {self.num_key_value_heads} (GQA ratio: {self.gqa_ratio}:1)")
        print(f"Layers: {self.num_hidden_layers}")
        print(f"Head Dimension: {self.head_dim}")
        
    def calculate_attention_flops(self, chunk_size: int, seq_len: int) -> int:
        """estimate attention FLOPs"""
        # Q projection
        q_flops = chunk_size * self.hidden_size * self.hidden_size
        
        # K,V projections (考虑GQA)
        kv_dim = self.hidden_size // self.gqa_ratio
        kv_flops = chunk_size * self.hidden_size * kv_dim * 2
        
        # Attention scores: Q @ K^T
        score_flops = chunk_size * seq_len * self.hidden_size
        
        # Attention @ V
        value_flops = chunk_size * seq_len * self.hidden_size
        
        # Output projection
        out_flops = chunk_size * self.hidden_size * self.hidden_size
        
        total = q_flops + kv_flops + score_flops + value_flops + out_flops

        attn_flops = score_flops + value_flops
        proj_flops = q_flops + kv_flops + out_flops
        return proj_flops, attn_flops
    
    def calculate_ffn_flops(self, chunk_size: int) -> int:
        """FFN FLOPs"""
        # Gate and Up projections (SwiGLU)
        up_gate_flops = chunk_size * self.hidden_size * self.intermediate_size * 2
        
        # Down projection
        down_flops = chunk_size * self.intermediate_size * self.hidden_size
        
        total = up_gate_flops + down_flops
        return total
    
    def calculate_other_flops(self, chunk_size: int) -> int:
        """other FLOPs（LayerNorm, residual and so on）"""
        # RMSNorm/LayerNorm (2 per layer)
        norm_flops = chunk_size * self.hidden_size * 4
        
        # Residual connections
        residual_flops = chunk_size * self.hidden_size * 2
        
        return norm_flops + residual_flops
    
    def estimate_p(self, chunk_size: int, kv_cache_len: int) -> float:
        """
        estimate p value（non attention/attention）
        
        Args:
            chunk_size: current chunk size
            kv_cache_len: KV cache长度（processed number of tokens）
        
        Returns:
            p值
        """
        seq_len = kv_cache_len + chunk_size
        
        proj_flops, attn_flops = self.calculate_attention_flops(chunk_size, seq_len)
        ffn_flops = self.calculate_ffn_flops(chunk_size)
        other_flops = self.calculate_other_flops(chunk_size)
        
        non_attn_flops = ffn_flops + other_flops + proj_flops
        
        return non_attn_flops / attn_flops
    
    def estimate_p_times_c(self, chunk_size: int, kv_cache_len: int) -> float:
        """estimate p * chunk_size（more stable estimation using chunk）"""
        return self.estimate_p(chunk_size, kv_cache_len) * chunk_size
    
    def get_p_table(self, chunk_sizes: List[int] = None, 
                    kv_cache_lens: List[int] = None) -> Dict:
        """
        cal different p values under different chun size
        
        Returns:
            dict contains p value
        """
        if chunk_sizes is None:
            chunk_sizes = [512, 1024, 2048, 4096]
        
        if kv_cache_lens is None:
            kv_cache_lens = [0, 2048, 8192, 16384, 32768, 65536, 98304]
        
        table = {}
        for chunk_size in chunk_sizes:
            table[chunk_size] = {}
            for kv_len in kv_cache_lens:
                p = self.estimate_p(chunk_size, kv_len)
                table[chunk_size][kv_len] = p
        
        return table
    
    def print_p_table(self, chunk_sizes: List[int] = None, 
                      kv_cache_lens: List[int] = None):
        """print p table"""
        table = self.get_p_table(chunk_sizes, kv_cache_lens)
        
        # head
        print("\nP-Value Table (non-attention / attention ratio)")
        print("=" * 80)
        print(f"{'KV Cache':<10}", end='')
        for chunk_size in sorted(table.keys()):
            print(f"C={chunk_size:<6}", end='')
        print()
        print("-" * 80)
        
        # data
        all_kv_lens = sorted(next(iter(table.values())).keys())
        for kv_len in all_kv_lens:
            print(f"{kv_len:<10}", end='')
            for chunk_size in sorted(table.keys()):
                p = table[chunk_size][kv_len]
                print(f"{p:8.2f}", end='')
            print()
    
    


def main():
    parser = argparse.ArgumentParser(description='Estimate p-values from model config')
    parser.add_argument('--config-path', type=str, help='Path to model config.json')
    parser.add_argument('--chunk-size', type=int, default=2048, 
                       help='Chunk size for estimation')
    parser.add_argument('--kv-cache-len', type=int, default=0,
                       help='KV cache length')
    parser.add_argument('--table', action='store_true',
                       help='Print p-value table')
    
    args = parser.parse_args()
    estimator = ModelPEstimator(config_path=args.config_path)
    estimator.print_model_info()
    
    p = estimator.estimate_p(args.chunk_size, args.kv_cache_len)
    p_c = estimator.estimate_p_times_c(args.chunk_size, args.kv_cache_len)
    
    # 打印p值表
    if args.table:
        estimator.print_p_table()

    print(f"p is {p}, p_c is {p_c}")
    

if __name__ == "__main__":
    main()