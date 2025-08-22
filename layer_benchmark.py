import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
from train_hybrid_llm import HybridConfig, ImprovedSSM, SimpleAttention

@dataclass
class BenchmarkConfig:
    hidden_size: int = 768
    num_heads: int = 12
    ssm_state_size: int = 32
    conv_kernel: int = 4
    expand_factor: int = 2
    dropout: float = 0.0  # No dropout for benchmarking
    batch_size: int = 8
    
    def __post_init__(self):
        self.intermediate_size = self.expand_factor * self.hidden_size


class LayerBenchmark:
    def __init__(self, config: BenchmarkConfig, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device)
        
        # Convert to HybridConfig for compatibility
        self.hybrid_config = HybridConfig(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            ssm_state_size=config.ssm_state_size,
            conv_kernel=config.conv_kernel,
            expand_factor=config.expand_factor,
            dropout=config.dropout
        )
        
        # Initialize layers
        self.mamba_layer = ImprovedSSM(self.hybrid_config).to(self.device)
        self.attention_layer = SimpleAttention(self.hybrid_config).to(self.device)
        
        # Set to eval mode for consistent benchmarking
        self.mamba_layer.eval()
        self.attention_layer.eval()
        
        print(f"🔧 Benchmark Setup:")
        print(f"   Device: {self.device}")
        print(f"   Hidden Size: {config.hidden_size}")
        print(f"   Heads: {config.num_heads}")
        print(f"   SSM State Size: {config.ssm_state_size}")
        print(f"   Batch Size: {config.batch_size}")
        print()
    
    def count_parameters(self, layer: nn.Module) -> int:
        """Count trainable parameters in a layer"""
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)
    
    def measure_memory(self, layer: nn.Module, seq_len: int) -> float:
        """Measure peak memory usage in MB"""
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create input
        x = torch.randn(self.config.batch_size, seq_len, self.config.hidden_size, 
                       device=self.device, requires_grad=True)
        
        # Forward pass
        with torch.no_grad():
            output = layer(x)
        
        # Get peak memory
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        
        # Cleanup
        del x, output
        torch.cuda.empty_cache()
        
        return peak_memory_mb
    
    def measure_latency(self, layer: nn.Module, seq_len: int, num_warmup: int = 10, num_runs: int = 100) -> Tuple[float, float]:
        """Measure forward pass latency (mean, std) in milliseconds"""
        # Create input
        x = torch.randn(self.config.batch_size, seq_len, self.config.hidden_size, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = layer(x)
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = layer(x)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Cleanup
        del x
        
        return np.mean(times), np.std(times)
    
    def measure_throughput(self, layer: nn.Module, seq_len: int, duration_seconds: float = 2.0) -> float:
        """Measure throughput (tokens/second)"""
        x = torch.randn(self.config.batch_size, seq_len, self.config.hidden_size, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = layer(x)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        iterations = 0
        
        with torch.no_grad():
            while time.perf_counter() - start_time < duration_seconds:
                _ = layer(x)
                iterations += 1
        
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time
        
        total_tokens = iterations * self.config.batch_size * seq_len
        throughput = total_tokens / elapsed_time
        
        # Cleanup
        del x
        
        return throughput
    
    def benchmark_sequence_lengths(self, seq_lengths: List[int]) -> Dict:
        """Benchmark both layers across different sequence lengths"""
        results = {
            'seq_lengths': seq_lengths,
            'mamba': {
                'latency_mean': [],
                'latency_std': [],
                'memory_mb': [],
                'throughput': [],
                'parameters': self.count_parameters(self.mamba_layer)
            },
            'attention': {
                'latency_mean': [],
                'latency_std': [],
                'memory_mb': [],
                'throughput': [],
                'parameters': self.count_parameters(self.attention_layer)
            }
        }
        
        print("🔍 Benchmarking Mamba vs Attention Layers")
        print("=" * 80)
        print(f"{'Seq Len':<8} {'Layer':<10} {'Latency (ms)':<15} {'Memory (MB)':<12} {'Throughput (tok/s)':<18} {'Params':<10}")
        print("-" * 80)
        
        for seq_len in seq_lengths:
            # Benchmark Mamba
            try:
                mamba_lat_mean, mamba_lat_std = self.measure_latency(self.mamba_layer, seq_len)
                mamba_memory = self.measure_memory(self.mamba_layer, seq_len)
                mamba_throughput = self.measure_throughput(self.mamba_layer, seq_len)
                
                results['mamba']['latency_mean'].append(mamba_lat_mean)
                results['mamba']['latency_std'].append(mamba_lat_std)
                results['mamba']['memory_mb'].append(mamba_memory)
                results['mamba']['throughput'].append(mamba_throughput)
                
                print(f"{seq_len:<8} {'Mamba':<10} {mamba_lat_mean:.2f}±{mamba_lat_std:.2f}{'':>4} {mamba_memory:.1f}{'':>7} {mamba_throughput:.0f}{'':>12} {results['mamba']['parameters']:,}")
                
            except RuntimeError as e:
                print(f"{seq_len:<8} {'Mamba':<10} {'OOM':>15} {'OOM':>12} {'OOM':>18} {results['mamba']['parameters']:,}")
                results['mamba']['latency_mean'].append(float('inf'))
                results['mamba']['latency_std'].append(0)
                results['mamba']['memory_mb'].append(float('inf'))
                results['mamba']['throughput'].append(0)
            
            # Benchmark Attention
            try:
                attn_lat_mean, attn_lat_std = self.measure_latency(self.attention_layer, seq_len)
                attn_memory = self.measure_memory(self.attention_layer, seq_len)
                attn_throughput = self.measure_throughput(self.attention_layer, seq_len)
                
                results['attention']['latency_mean'].append(attn_lat_mean)
                results['attention']['latency_std'].append(attn_lat_std)
                results['attention']['memory_mb'].append(attn_memory)
                results['attention']['throughput'].append(attn_throughput)
                
                print(f"{seq_len:<8} {'Attention':<10} {attn_lat_mean:.2f}±{attn_lat_std:.2f}{'':>4} {attn_memory:.1f}{'':>7} {attn_throughput:.0f}{'':>12} {results['attention']['parameters']:,}")
                
            except RuntimeError as e:
                print(f"{seq_len:<8} {'Attention':<10} {'OOM':>15} {'OOM':>12} {'OOM':>18} {results['attention']['parameters']:,}")
                results['attention']['latency_mean'].append(float('inf'))
                results['attention']['latency_std'].append(0)
                results['attention']['memory_mb'].append(float('inf'))
                results['attention']['throughput'].append(0)
            
            # Add separator for readability
            if seq_len != seq_lengths[-1]:
                print()
            
            # Force garbage collection between sequence lengths
            gc.collect()
            torch.cuda.empty_cache()
        
        print("=" * 80)
        
        return results
    
    def print_summary(self, results: Dict):
        """Print a summary of the benchmark results"""
        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 50)
        
        # Parameter comparison
        mamba_params = results['mamba']['parameters']
        attn_params = results['attention']['parameters']
        print(f"Parameters:")
        print(f"  Mamba:     {mamba_params:,}")
        print(f"  Attention: {attn_params:,}")
        print(f"  Ratio:     {mamba_params/attn_params:.2f}x")
        print()
        
        # Find crossover points and analyze scaling
        seq_lengths = results['seq_lengths']
        
        # Latency analysis
        print("🚀 Performance Analysis:")
        mamba_faster_count = 0
        for i, seq_len in enumerate(seq_lengths):
            mamba_lat = results['mamba']['latency_mean'][i]
            attn_lat = results['attention']['latency_mean'][i]
            
            if mamba_lat < attn_lat and mamba_lat != float('inf'):
                mamba_faster_count += 1
        
        print(f"  Mamba faster in {mamba_faster_count}/{len(seq_lengths)} cases")
        
        # Memory analysis
        print("\n💾 Memory Analysis:")
        mamba_mem_efficient_count = 0
        for i, seq_len in enumerate(seq_lengths):
            mamba_mem = results['mamba']['memory_mb'][i]
            attn_mem = results['attention']['memory_mb'][i]
            
            if mamba_mem < attn_mem and mamba_mem != float('inf'):
                mamba_mem_efficient_count += 1
        
        print(f"  Mamba more memory efficient in {mamba_mem_efficient_count}/{len(seq_lengths)} cases")
        
        # Scaling analysis
        print("\n📈 Scaling Analysis:")
        valid_indices = [i for i, (m_lat, a_lat) in enumerate(zip(results['mamba']['latency_mean'], results['attention']['latency_mean'])) 
                        if m_lat != float('inf') and a_lat != float('inf')]
        
        if len(valid_indices) >= 2:
            # Compare first vs last valid measurements
            first_idx, last_idx = valid_indices[0], valid_indices[-1]
            first_seq, last_seq = seq_lengths[first_idx], seq_lengths[last_idx]
            
            mamba_scale = results['mamba']['latency_mean'][last_idx] / results['mamba']['latency_mean'][first_idx]
            attn_scale = results['attention']['latency_mean'][last_idx] / results['attention']['latency_mean'][first_idx]
            seq_scale = last_seq / first_seq
            
            print(f"  Sequence length scale: {seq_scale:.1f}x ({first_seq} → {last_seq})")
            print(f"  Mamba latency scale:   {mamba_scale:.1f}x")
            print(f"  Attention latency scale: {attn_scale:.1f}x")
            print(f"  Mamba scaling efficiency: {seq_scale/mamba_scale:.2f}")
            print(f"  Attention scaling efficiency: {seq_scale/attn_scale:.2f}")


def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This benchmark requires GPU.")
        return
    
    print("🚀 Mamba vs Attention Layer Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # Configuration
    config = BenchmarkConfig(
        hidden_size=768,
        batch_size=8,  # Reasonable batch size for benchmarking
    )
    
    # Sequence lengths to test (powers of 2 for clean scaling analysis)
    seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    # Initialize benchmark
    benchmark = LayerBenchmark(config)
    
    # Run benchmark
    try:
        results = benchmark.benchmark_sequence_lengths(seq_lengths)
        benchmark.print_summary(results)
        
        # Optionally save results
        print(f"\n💾 Results saved to benchmark_results.json")
        import json
        with open('benchmark_results.json', 'w') as f:
            # Convert any inf values to strings for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, list):
                            serializable_results[key][k] = [str(x) if x == float('inf') else x for x in v]
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
    
    finally:
        # Cleanup
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
