#!/usr/bin/env python3
"""
Optimized Learning Rate Training for SpectralLLM
===============================================

Advanced training script incorporating optimizations from train_optimized_lr.py:
- Cosine annealing LR scheduler with proper warmup
- Better optimizer parameters (betas) for language models
- Target perplexity tracking and early stopping
- Enhanced progress monitoring with LR tracking
- Comprehensive hypothesis testing framework
"""

import os
import sys
import argparse
import logging
import json
import time
import math
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# SpectralLLM imports
import spectralllm
from spectralllm.training.trainer import SpectralTrainer, EnhancedTextDataset, log_exception
from spectralllm.core.config import Config

# Check for MPS optimizations
try:
    from spectralllm.utils.mps_optimizations import optimize_for_mps, setup_mps_optimizations
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    MPS_AVAILABLE = False


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup enhanced logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file or "optimized_lr_training.log")
        ]
    )
    return logging.getLogger(__name__)


def create_sample_dataset(args) -> List[str]:
    """Create diverse sample dataset for testing"""
    patterns = [
        "Deep learning has revolutionized artificial intelligence through neural networks. ",
        "Machine learning algorithms can learn complex patterns from large datasets. ",
        "Natural language processing enables computers to understand human language. ",
        "Transformer architectures have become the foundation of modern language models. ",
        "Wavelet analysis provides multi-resolution decomposition of signals and data. ",
        "Spectral methods offer efficient computation for sequence modeling tasks. ",
        "Optimization algorithms like Adam adapt learning rates during training. ",
        "Mathematics forms the theoretical foundation of machine learning advances. ",
        "Gradient descent iteratively minimizes loss functions in neural networks. ",
        "Attention mechanisms allow models to focus on relevant parts of input. ",
    ]
    
    texts = []
    for i in range(args.num_samples):
        # Create text of varying length
        sample_length = torch.randint(100, 300, (1,)).item()
        sample_text = ""
        
        while len(sample_text.split()) < sample_length:
            pattern = patterns[i % len(patterns)]
            sample_text += pattern
        
        texts.append(sample_text)
    
    return texts


def run_lr_experiment(args):
    """
    Run learning rate optimization experiment.
    Tests hypothesis that optimized LR will achieve target perplexity.
    """
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Log experiment setup
    logger.info("🚀 OPTIMIZED LEARNING RATE EXPERIMENT")
    logger.info(f"📊 Testing hypothesis: LR {args.learning_rate} will achieve sub-{args.target_perplexity} perplexity")
    logger.info(f"🆚 Comparison: Standard training typically uses 1e-6 LR ({1e-6 / args.learning_rate:.0f}x smaller)")
    logger.info(f"🎯 Target perplexity: {args.target_perplexity}")
    logger.info("=" * 60)
    
    try:
        # Device setup with MPS support
        if args.use_mps and MPS_AVAILABLE:
            device = torch.device('mps')
            logger.info("Using Apple Silicon MPS")
            setup_mps_optimizations()
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Using CUDA GPU")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Create output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(args.output_dir, f"lr_experiment_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create or load dataset
        if args.dataset_path and os.path.exists(args.dataset_path):
            logger.info(f"Loading dataset from {args.dataset_path}")
            with open(args.dataset_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
        else:
            logger.info("Creating sample dataset")
            texts = create_sample_dataset(args)
        
        logger.info(f"Dataset contains {len(texts)} texts")
        
        # Create tokenizer
        tokenizer = spectralllm.SimpleTokenizer(mode='char')
        tokenizer.build_vocab(texts, max_vocab_size=args.vocab_size)
        vocab_size = tokenizer.get_vocab_size()
        logger.info(f"Vocabulary size: {vocab_size}")
        
        # Split dataset
        train_size = int(len(texts) * 0.8)
        val_size = int(len(texts) * 0.1)
        
        train_texts = texts[:train_size]
        val_texts = texts[train_size:train_size + val_size]
        
        # Create enhanced datasets with non-overlapping sequences
        train_dataset = EnhancedTextDataset(
            train_texts,
            tokenizer,
            seq_length=args.seq_length,
            stride=args.seq_length,  # Non-overlapping
            validate_sequences=True
        )
        
        val_dataset = EnhancedTextDataset(
            val_texts,
            tokenizer,
            seq_length=args.seq_length,
            stride=args.seq_length,  # Non-overlapping
            validate_sequences=True
        )
        
        # Log dataset statistics
        train_stats = train_dataset.get_dataset_stats()
        logger.info(f"Train dataset: {train_stats}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # Create model configuration
        config = Config(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            max_seq_length=args.seq_length,
            dropout=args.dropout,
            use_signal_embed=args.use_signal_embed,
            use_wavelet_attention=args.use_wavelet_attention,
            harmonic_bases=args.harmonic_bases,
            wavelet_type=args.wavelet_type,
            wavelet_levels=args.wavelet_levels,
            use_adaptive_basis=args.use_adaptive_basis,
            wavelet_families=['db4', 'sym4', 'dmey'],
            use_fourier_convolution=args.use_fourier_convolution,
            use_stochastic_transform=args.use_stochastic_transform,
            spectral_gap_analysis=args.spectral_gap_analysis,
            # Optimized training parameters
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_hrfevo=args.enable_evolution,
            evolution_population=args.evolution_population,
            evolution_generations=args.evolution_generations
        )
        
        # Create model
        logger.info("Creating SpectralLLM model...")
        model = spectralllm.SpectralLLM(config)
        logger.info(f"Model parameters: {model.count_parameters():,}")
        
        # Apply MPS optimizations if available
        if args.use_mps and MPS_AVAILABLE:
            logger.info("Applying MPS optimizations...")
            model = optimize_for_mps(model, device)
        
        # Move model to device
        model = model.to(device)
        
        # Create trainer with optimized settings
        trainer = SpectralTrainer(model, config, use_optimized_lr=True)
        
        # Log optimization details
        logger.info(f"🎬 Training Configuration:")
        logger.info(f"   Learning Rate: {args.learning_rate} (optimized)")
        logger.info(f"   Warmup Steps: {args.warmup_steps}")
        logger.info(f"   Min LR Ratio: {args.min_lr_ratio} ({args.min_lr_ratio*100}% of peak)")
        logger.info(f"   Optimizer Betas: (0.9, 0.95) for language models")
        logger.info(f"   Scheduler: Cosine annealing with warmup")
        
        # Test forward pass
        logger.info("Testing forward pass...")
        try:
            model.eval()
            with torch.no_grad():
                test_input = torch.randint(0, vocab_size, (2, args.seq_length), device=device)
                test_output = model(test_input)
                logger.info(f"✅ Test forward pass successful: {test_output.shape}")
        except Exception as e:
            error_msg = log_exception(e, "test forward pass")
            logger.error(f"❌ Test forward pass failed: {error_msg}")
            return
        
        # Save experiment configuration
        experiment_config = {
            'model_config': config.to_dict(),
            'training_args': vars(args),
            'experiment_timestamp': timestamp,
            'hypothesis': f"LR {args.learning_rate} will achieve sub-{args.target_perplexity} perplexity",
            'device': str(device),
            'mps_optimizations': args.use_mps and MPS_AVAILABLE
        }
        
        config_path = os.path.join(experiment_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        # Run training with target perplexity tracking
        logger.info("=" * 60)
        logger.info("🎬 STARTING OPTIMIZED TRAINING")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        history = trainer.train(
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            output_dir=experiment_dir,
            target_perplexity=args.target_perplexity,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio
        )
        
        training_time = time.time() - start_time
        
        # Analyze results
        best_train_ppl = min(history['train_perplexity'])
        best_val_ppl = min(history['val_perplexity']) if history['val_perplexity'] else float('inf')
        
        logger.info("=" * 60)
        logger.info("🏆 EXPERIMENT RESULTS")
        logger.info("=" * 60)
        logger.info(f"📈 Best training perplexity: {best_train_ppl:.2f}")
        logger.info(f"📈 Best validation perplexity: {best_val_ppl:.2f}")
        logger.info(f"🎯 Target perplexity: {args.target_perplexity}")
        logger.info(f"✅ Training target achieved: {'YES' if best_train_ppl < args.target_perplexity else 'NO'}")
        logger.info(f"✅ Validation target achieved: {'YES' if best_val_ppl < args.target_perplexity else 'NO'}")
        logger.info(f"⏱️  Total training time: {training_time:.1f}s")
        logger.info(f"🔢 Total errors: {trainer.error_count}")
        logger.info(f"📁 Results saved to: {experiment_dir}")
        
        # Save final results
        results = {
            'experiment_config': experiment_config,
            'training_history': history,
            'best_train_perplexity': best_train_ppl,
            'best_val_perplexity': best_val_ppl,
            'target_perplexity': args.target_perplexity,
            'target_achieved_train': best_train_ppl < args.target_perplexity,
            'target_achieved_val': best_val_ppl < args.target_perplexity,
            'training_time_seconds': training_time,
            'total_errors': trainer.error_count,
            'total_batches': trainer.batch_count
        }
        
        results_path = os.path.join(experiment_dir, "experiment_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate sample text to verify model quality
        logger.info("🤖 Generating sample text...")
        try:
            model.eval()
            with torch.no_grad():
                prompt = "The future of artificial intelligence"
                prompt_tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([prompt_tokens[:min(len(prompt_tokens), args.seq_length//2)]], device=device)
                
                generated_tokens = []
                for _ in range(50):
                    outputs = model(input_ids)
                    next_token_logits = outputs[0, -1, :]
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    generated_tokens.append(next_token)
                    
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
                    
                    if input_ids.size(1) >= args.seq_length:
                        break
                
                full_tokens = input_ids[0].cpu().tolist()
                generated_text = tokenizer.decode(full_tokens)
                logger.info(f"Generated text: {generated_text}")
                
                # Save generated text
                with open(os.path.join(experiment_dir, "generated_sample.txt"), 'w') as f:
                    f.write(generated_text)
        
        except Exception as e:
            log_exception(e, "text generation")
            logger.warning("Could not generate sample text")
        
        return results
    
    except Exception as e:
        error_msg = log_exception(e, "optimized LR experiment")
        logger.critical(f"💥 Experiment failed: {error_msg}")
        return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Optimized Learning Rate Training for SpectralLLM")
    
    # Experiment parameters
    parser.add_argument("--target_perplexity", type=float, default=5.0,
                       help="Target perplexity to test hypothesis")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Optimized learning rate (vs 1e-6 typically used)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                       help="Minimum LR as ratio of peak (maintains 10%% of peak)")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Shorter warmup for faster convergence")
    
    # Data parameters
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to text dataset file")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="Number of sample texts if no dataset provided")
    parser.add_argument("--vocab_size", type=int, default=1000,
                       help="Maximum vocabulary size")
    parser.add_argument("--seq_length", type=int, default=256,
                       help="Sequence length for training")
    
    # Model parameters
    parser.add_argument("--embed_dim", type=int, default=256,
                       help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=1024,
                       help="Hidden layer dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    
    # SpectralLLM features
    parser.add_argument("--use_signal_embed", action="store_true",
                       help="Use signal embedding")
    parser.add_argument("--use_wavelet_attention", action="store_true",
                       help="Use wavelet-based attention")
    parser.add_argument("--harmonic_bases", type=int, default=32,
                       help="Number of harmonic bases")
    parser.add_argument("--wavelet_type", type=str, default="db4",
                       help="Wavelet type")
    parser.add_argument("--wavelet_levels", type=int, default=3,
                       help="Number of wavelet levels")
    parser.add_argument("--use_adaptive_basis", action="store_true",
                       help="Use adaptive basis selection")
    parser.add_argument("--use_fourier_convolution", action="store_true",
                       help="Use Fourier convolution attention")
    parser.add_argument("--use_stochastic_transform", action="store_true",
                       help="Use stochastic wavelet transform")
    parser.add_argument("--spectral_gap_analysis", action="store_true",
                       help="Enable spectral gap analysis")
    
    # Evolution parameters
    parser.add_argument("--enable_evolution", action="store_true",
                       help="Enable HRFEvo during training")
    parser.add_argument("--evolution_population", type=int, default=10,
                       help="Evolution population size")
    parser.add_argument("--evolution_generations", type=int, default=5,
                       help="Number of evolution generations")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of data loader workers")
    
    # Device and optimization
    parser.add_argument("--use_mps", action="store_true",
                       help="Use MPS optimizations if available")
    
    # Logging and output
    parser.add_argument("--output_dir", type=str, default="./lr_experiments",
                       help="Output directory for experiment results")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("🚀 OPTIMIZED LEARNING RATE EXPERIMENT FOR SPECTRALLLM")
    print(f"📊 Testing hypothesis: LR {args.learning_rate} will achieve sub-{args.target_perplexity} perplexity")
    print(f"🔬 Optimizations: Cosine annealing, better betas, proper warmup")
    print("=" * 70)
    
    results = run_lr_experiment(args)
    
    if results:
        print("=" * 70)
        print(f"🏆 EXPERIMENT COMPLETE!")
        print(f"📈 Best perplexity: {results['best_val_perplexity']:.2f}")
        print(f"🎯 Target: {results['target_perplexity']}")
        print(f"✅ Success: {'YES' if results['target_achieved_val'] else 'NO'}")
        print(f"⏱️  Time: {results['training_time_seconds']:.1f}s")
        print("=" * 70)
    else:
        print("❌ Experiment failed - check logs for details") 