import torch
import torch.nn as nn
import time
import gc
import json
import os
from typing import Dict, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import psutil

# Import our models
from ar_llm import MinimalLLM, ModelConfig, convert_to_hybrid_config, evaluate_model as evaluate_ar_model
from train_hybrid_llm import HybridModel, HybridConfig
from shared_data import shared_data_manager

@dataclass
class BenchmarkResults:
    """Container for benchmark results"""
    model_name: str
    parameters: int
    memory_peak_mb: float
    memory_allocated_mb: float
    training_time_seconds: float
    throughput_tokens_per_second: float
    final_val_loss: float
    final_val_perplexity: float
    final_val_accuracy: float
    steps_completed: int
    
    def to_dict(self):
        return asdict(self)


class ModelBenchmark:
    def __init__(self, device='cuda', steps=1000):
        self.device = torch.device(device)
        self.steps = steps
        print(f"🔧 Model Benchmark Setup")
        print(f"   Device: {self.device}")
        print(f"   Steps: {self.steps}")
        print(f"   GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}")
        print()
    
    def get_memory_stats(self) -> Tuple[float, float]:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            return allocated, peak
        return 0, 0
    
    def reset_memory_stats(self):
        """Reset memory tracking"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
    
    def benchmark_ar_model(self, config: ModelConfig, train_loader, val_loader) -> BenchmarkResults:
        """Benchmark the AR (pure attention) model"""
        print("🚀 Benchmarking AR Model (Pure Attention)")
        
        self.reset_memory_stats()
        
        # Create model
        model = MinimalLLM(config).to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params:,}")
        
        # Import and setup Muon optimizer (as in ar_llm.py)
        from ar_llm import setup_muon_optimizer
        
        # Setup optimizers (original Muon setup)
        optimizers = setup_muon_optimizer(model, config)
        
        # Learning rate schedule
        schedulers = []
        for optimizer in optimizers:
            warmup_steps = self.steps // 20
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (self.steps - warmup_steps)
                    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            schedulers.append(scheduler)
        scaler = torch.cuda.amp.GradScaler()
        
        # Training loop
        model.train()
        start_time = time.time()
        total_tokens = 0
        step = 0
        
        train_iter = iter(train_loader)
        pbar = tqdm(total=self.steps, desc="AR Training", leave=False)
        
        while step < self.steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            # Handle data format
            if isinstance(batch, tuple):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
            else:
                x = batch.to(self.device)
                y = x[:, 1:].clone()
                x = x[:, :-1]
            
            total_tokens += y.numel()
            
            # Forward pass
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.view(-1, config.vocab_size), y.contiguous().view(-1))
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Handle multiple optimizers (Muon setup)
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            for optimizer in optimizers:
                scaler.step(optimizer)
                optimizer.zero_grad()
            for scheduler in schedulers:
                scheduler.step()
            scaler.update()
            
            step += 1
            pbar.update(1)
            
            if step % 100 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()
        
        training_time = time.time() - start_time
        throughput = total_tokens / training_time
        
        # Memory stats
        memory_allocated, memory_peak = self.get_memory_stats()
        
        # Final evaluation
        print("   Evaluating...")
        eval_results = evaluate_ar_model(model, val_loader, config)
        
        # Cleanup
        del model, optimizers, schedulers, scaler
        self.reset_memory_stats()
        
        return BenchmarkResults(
            model_name="AR_Model",
            parameters=total_params,
            memory_peak_mb=memory_peak,
            memory_allocated_mb=memory_allocated,
            training_time_seconds=training_time,
            throughput_tokens_per_second=throughput,
            final_val_loss=eval_results['val_loss'],
            final_val_perplexity=eval_results['val_perplexity'],
            final_val_accuracy=eval_results['val_accuracy'],
            steps_completed=step
        )
    
    def benchmark_hybrid_model(self, config: HybridConfig, train_loader, val_loader) -> BenchmarkResults:
        """Benchmark the Hybrid model"""
        print("🔀 Benchmarking Hybrid Model (Mamba + Attention)")
        
        self.reset_memory_stats()
        
        # Create model
        model = HybridModel(config).to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params:,}")
        print(f"   Pattern: {config.layer_pattern}")
        
        # Setup optimizer (same as hybrid training)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,  # Default from hybrid
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Simple scheduler
        warmup_steps = self.steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (self.steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.cuda.amp.GradScaler()
        
        # Training loop
        model.train()
        start_time = time.time()
        total_tokens = 0
        step = 0
        
        train_iter = iter(train_loader)
        pbar = tqdm(total=self.steps, desc="Hybrid Training", leave=False)
        
        while step < self.steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            batch = batch.to(self.device)
            total_tokens += batch.numel()
            
            # Forward pass
            with torch.cuda.amp.autocast():
                _, loss = model(batch, labels=batch)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            step += 1
            pbar.update(1)
            
            if step % 100 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()
        
        training_time = time.time() - start_time
        throughput = total_tokens / training_time
        
        # Memory stats
        memory_allocated, memory_peak = self.get_memory_stats()
        
        # Final evaluation (simplified)
        print("   Evaluating...")
        model.eval()
        total_loss = 0
        total_tokens_eval = 0
        total_correct = 0
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 100:  # Limit eval batches
                    break
                batch = batch.to(self.device)
                
                with torch.cuda.amp.autocast():
                    logits, _ = model(batch)
                    
                    # Compute loss manually
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch[..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, config.vocab_size),
                        shift_labels.view(-1)
                    )
                
                total_loss += loss.item() * shift_labels.numel()
                total_tokens_eval += shift_labels.numel()
                
                predictions = shift_logits.argmax(dim=-1)
                total_correct += (predictions == shift_labels).sum().item()
        
        avg_loss = total_loss / max(total_tokens_eval, 1)
        accuracy = total_correct / max(total_tokens_eval, 1)
        perplexity = torch.exp(torch.tensor(min(avg_loss, 10))).item()
        
        # Cleanup
        del model, optimizer, scheduler, scaler
        self.reset_memory_stats()
        
        return BenchmarkResults(
            model_name="Hybrid_Model",
            parameters=total_params,
            memory_peak_mb=memory_peak,
            memory_allocated_mb=memory_allocated,
            training_time_seconds=training_time,
            throughput_tokens_per_second=throughput,
            final_val_loss=avg_loss,
            final_val_perplexity=perplexity,
            final_val_accuracy=accuracy,
            steps_completed=step
        )
    
    def run_comparison(self, steps=1000):
        """Run full comparison benchmark"""
        self.steps = steps
        
        print("🏁 Starting Model Comparison Benchmark")
        print("=" * 60)
        
        # Setup configs to match
        ar_config = ModelConfig()
        ar_config.max_steps = steps
        
        hybrid_config = convert_to_hybrid_config(ar_config)
        hybrid_config.num_steps = steps
        
        print(f"📊 Configurations:")
        print(f"   Steps: {steps}")
        print(f"   Hidden Size: {ar_config.d_model}")
        print(f"   Layers: {ar_config.n_layers}")
        print(f"   Heads: {ar_config.n_heads}")
        print(f"   Batch Size: {ar_config.batch_size}")
        print()
        
        # Load shared data
        print("📚 Loading shared dataset...")
        train_loader, val_loader = shared_data_manager.load_or_create_datasets(hybrid_config)
        tokenizer = shared_data_manager.get_tokenizer()
        
        ar_config.vocab_size = tokenizer.vocab_size
        hybrid_config.vocab_size = tokenizer.vocab_size
        
        print(f"   Dataset: {len(train_loader)} train batches, {len(val_loader)} val batches")
        print(f"   Vocab size: {tokenizer.vocab_size:,}")
        print()
        
        # Benchmark AR model
        ar_results = self.benchmark_ar_model(ar_config, train_loader, val_loader)
        print(f"✅ AR Model completed: {ar_results.training_time_seconds:.1f}s")
        
        # Wait a bit and clean up
        time.sleep(2)
        self.reset_memory_stats()
        
        # Benchmark Hybrid model
        hybrid_results = self.benchmark_hybrid_model(hybrid_config, train_loader, val_loader)
        print(f"✅ Hybrid Model completed: {hybrid_results.training_time_seconds:.1f}s")
        
        # Compare results
        self.print_comparison(ar_results, hybrid_results)
        
        # Save results
        results = {
            'ar_model': ar_results.to_dict(),
            'hybrid_model': hybrid_results.to_dict(),
            'comparison': self.compute_comparison_metrics(ar_results, hybrid_results)
        }
        
        with open('model_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to model_comparison_results.json")
        
        return ar_results, hybrid_results
    
    def compute_comparison_metrics(self, ar_results: BenchmarkResults, hybrid_results: BenchmarkResults) -> Dict:
        """Compute comparison metrics"""
        return {
            'speed_ratio': ar_results.training_time_seconds / hybrid_results.training_time_seconds,
            'throughput_ratio': ar_results.throughput_tokens_per_second / hybrid_results.throughput_tokens_per_second,
            'memory_ratio': ar_results.memory_peak_mb / hybrid_results.memory_peak_mb,
            'params_ratio': ar_results.parameters / hybrid_results.parameters,
            'loss_difference': ar_results.final_val_loss - hybrid_results.final_val_loss,
            'perplexity_difference': ar_results.final_val_perplexity - hybrid_results.final_val_perplexity,
            'accuracy_difference': ar_results.final_val_accuracy - hybrid_results.final_val_accuracy
        }
    
    def print_comparison(self, ar_results: BenchmarkResults, hybrid_results: BenchmarkResults):
        """Print detailed comparison"""
        print("\n" + "=" * 80)
        print("📊 MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        # Performance metrics
        print(f"{'Metric':<25} {'AR Model':<15} {'Hybrid Model':<15} {'Ratio':<10}")
        print("-" * 70)
        print(f"{'Parameters':<25} {ar_results.parameters:<15,} {hybrid_results.parameters:<15,} {ar_results.parameters/hybrid_results.parameters:<10.2f}")
        print(f"{'Training Time (s)':<25} {ar_results.training_time_seconds:<15.1f} {hybrid_results.training_time_seconds:<15.1f} {ar_results.training_time_seconds/hybrid_results.training_time_seconds:<10.2f}")
        print(f"{'Throughput (tok/s)':<25} {ar_results.throughput_tokens_per_second:<15.0f} {hybrid_results.throughput_tokens_per_second:<15.0f} {ar_results.throughput_tokens_per_second/hybrid_results.throughput_tokens_per_second:<10.2f}")
        print(f"{'Peak Memory (MB)':<25} {ar_results.memory_peak_mb:<15.1f} {hybrid_results.memory_peak_mb:<15.1f} {ar_results.memory_peak_mb/hybrid_results.memory_peak_mb:<10.2f}")
        print()
        
        # Quality metrics
        print(f"{'Quality Metric':<25} {'AR Model':<15} {'Hybrid Model':<15} {'Difference':<10}")
        print("-" * 70)
        print(f"{'Validation Loss':<25} {ar_results.final_val_loss:<15.4f} {hybrid_results.final_val_loss:<15.4f} {ar_results.final_val_loss - hybrid_results.final_val_loss:<10.4f}")
        print(f"{'Validation PPL':<25} {ar_results.final_val_perplexity:<15.2f} {hybrid_results.final_val_perplexity:<15.2f} {ar_results.final_val_perplexity - hybrid_results.final_val_perplexity:<10.2f}")
        print(f"{'Validation Accuracy':<25} {ar_results.final_val_accuracy:<15.4f} {hybrid_results.final_val_accuracy:<15.4f} {ar_results.final_val_accuracy - hybrid_results.final_val_accuracy:<10.4f}")
        
        # Summary
        print("\n🏆 SUMMARY:")
        faster_model = "AR" if ar_results.training_time_seconds < hybrid_results.training_time_seconds else "Hybrid"
        better_model = "AR" if ar_results.final_val_loss < hybrid_results.final_val_loss else "Hybrid"
        
        print(f"   Faster Training: {faster_model} Model")
        print(f"   Better Final Loss: {better_model} Model")
        print(f"   Speed Difference: {abs(ar_results.training_time_seconds - hybrid_results.training_time_seconds):.1f}s")
        print(f"   Loss Difference: {abs(ar_results.final_val_loss - hybrid_results.final_val_loss):.4f}")


def main():
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This benchmark requires GPU.")
        return
    
    print("🚀 Model Comparison Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run benchmark
    benchmark = ModelBenchmark()
    
    # Quick benchmark (1000 steps)
    ar_results, hybrid_results = benchmark.run_comparison(steps=1000)
    
    print("\n🎉 Benchmark completed!")


if __name__ == "__main__":
    main()
