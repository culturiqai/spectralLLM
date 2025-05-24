#!/usr/bin/env python3
"""
Enhanced SpectralLLM Training Script
===================================

Comprehensive training script with advanced features:
- Enhanced error handling and logging
- MPS optimizations for Apple Silicon
- Better dataset validation
- Flag file system for remote control
- Memory tracking and debugging
- Proper metrics calculation
"""

import os
import sys
import argparse
import logging
import json
import time
import traceback
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# SpectralLLM imports
import spectralllm
from spectralllm.training.trainer import SpectralTrainer, EnhancedTextDataset, log_exception
from spectralllm.core.config import Config
from spectralllm.core.evolution import HRFEvoController

# Check for MPS optimizations
try:
    from spectralllm.utils.mps_optimizations import optimize_for_mps, setup_mps_optimizations
    MPS_AVAILABLE = torch.backends.mps.is_available()
    if MPS_AVAILABLE:
        print("✅ Apple Silicon MPS optimizations available")
except ImportError:
    MPS_AVAILABLE = False
    print("⚠️  MPS optimizations not available")

# Setup enhanced logging
def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup enhanced logging with file output"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file or "enhanced_training.log")
        ]
    )
    return logging.getLogger(__name__)


def create_sample_dataset(args) -> List[str]:
    """Create a sample dataset for demonstration"""
    texts = []
    
    # Create more diverse sample data
    patterns = [
        "The quick brown fox jumps over the lazy dog. ",
        "Machine learning is transforming the world of artificial intelligence. ",
        "Deep neural networks can learn complex patterns from data. ",
        "Transformers have revolutionized natural language processing. ",
        "Wavelet analysis provides multi-resolution signal decomposition. ",
        "Spectral methods offer efficient computation for sequence modeling. ",
        "Python is a powerful programming language for data science. ",
        "Mathematics forms the foundation of computer science algorithms. ",
    ]
    
    # Generate text samples of varying lengths
    for i in range(args.num_samples):
        # Create text of random length
        sample_length = torch.randint(50, 200, (1,)).item()
        sample_text = ""
        
        while len(sample_text.split()) < sample_length:
            pattern = patterns[i % len(patterns)]
            sample_text += pattern
        
        texts.append(sample_text)
    
    return texts


def train_enhanced_spectralllm(args):
    """Enhanced SpectralLLM training with comprehensive features"""
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting enhanced SpectralLLM training")
    
    try:
        # Determine device with MPS support
        if args.use_mps and MPS_AVAILABLE:
            device = torch.device('mps')
            logger.info("Using Apple Silicon MPS")
            # Setup MPS optimizations
            setup_mps_optimizations()
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Using CUDA GPU")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
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
        
        # Build vocabulary
        tokenizer.build_vocab(texts, max_vocab_size=args.vocab_size)
        vocab_size = tokenizer.get_vocab_size()
        logger.info(f"Vocabulary size: {vocab_size}")
        
        # Save tokenizer
        tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
        tokenizer.save_vocab(tokenizer_path)
        
        # Create enhanced datasets with validation
        train_size = int(len(texts) * 0.8)
        val_size = int(len(texts) * 0.1)
        
        train_texts = texts[:train_size]
        val_texts = texts[train_size:train_size + val_size]
        test_texts = texts[train_size + val_size:]
        
        # Use non-overlapping sequences for better training
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
        val_stats = val_dataset.get_dataset_stats()
        logger.info(f"Validation dataset: {val_stats}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == 'cuda'
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == 'cuda'
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
            # Training configuration
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
        
        # Apply MPS optimizations if available and requested
        if args.use_mps and MPS_AVAILABLE:
            logger.info("Applying MPS optimizations...")
            model = optimize_for_mps(model, device)
        
        # Move model to device
        model = model.to(device)
        
        # Create trainer with enhanced features
        trainer = SpectralTrainer(model, config, use_optimized_lr=args.use_optimized_lr)
        
        # Setup optimizer and scheduler with warmup
        steps_per_epoch = len(train_loader)
        total_steps = args.epochs * steps_per_epoch
        
        # Log training setup
        logger.info(f"Training setup:")
        logger.info(f"   Steps per epoch: {steps_per_epoch}")
        logger.info(f"   Total steps: {total_steps}")
        logger.info(f"   Optimized LR: {args.use_optimized_lr}")
        if args.use_optimized_lr:
            logger.info(f"   Warmup steps: {args.warmup_steps}")
            logger.info(f"   Min LR ratio: {args.min_lr_ratio}")
            logger.info(f"   Optimizer betas: (0.9, 0.95)")
        
        # For backward compatibility, keep the old scheduler setup for non-optimized mode
        if not args.use_optimized_lr:
            def lr_lambda(current_step):
                if current_step < args.warmup_steps:
                    return float(current_step) / float(max(1, args.warmup_steps))
                return max(
                    0.01,  # Minimum learning rate factor
                    float(total_steps - current_step) / 
                    float(max(1, total_steps - args.warmup_steps))
                )
            
            scheduler = LambdaLR(trainer.optimizer, lr_lambda)
            trainer.scheduler = scheduler
        
        # Initialize HRFEvo controller if enabled
        hrfevo_controller = None
        if args.enable_evolution:
            logger.info("Initializing HRFEvo controller...")
            hrfevo_controller = HRFEvoController(config)
            trainer.hrfevo_controller = hrfevo_controller
        
        # Load checkpoint if provided
        start_epoch = 0
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            logger.info(f"Loading checkpoint from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            trainer.train_losses = checkpoint.get('train_losses', [])
            trainer.val_losses = checkpoint.get('val_losses', [])
            logger.info(f"Resumed from epoch {start_epoch}")
        
        # Save configuration
        config_path = os.path.join(args.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Training loop with enhanced monitoring
        logger.info(f"Starting training for {args.epochs} epochs...")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        
        # Test forward pass before training
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
        
        # Main training loop
        history = trainer.train(
            train_loader, 
            val_loader, 
            num_epochs=args.epochs,
            output_dir=args.output_dir,
            target_perplexity=getattr(args, 'target_perplexity', None),
            warmup_steps=args.warmup_steps,
            min_lr_ratio=getattr(args, 'min_lr_ratio', 0.1)
        )
        
        # Final evaluation on test set if available
        if test_texts:
            logger.info("Running final evaluation...")
            test_dataset = EnhancedTextDataset(
                test_texts,
                tokenizer,
                seq_length=args.seq_length,
                stride=args.seq_length,
                validate_sequences=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
            
            test_metrics = trainer.evaluate(test_loader)
            logger.info(f"Test Results - Loss: {test_metrics['val_loss']:.4f}, "
                       f"PPL: {test_metrics['val_perplexity']:.2f}, "
                       f"Acc: {test_metrics['val_accuracy']:.4f}")
            
            # Save test results
            test_results_path = os.path.join(args.output_dir, "test_results.json")
            with open(test_results_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, "final_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'history': history,
            'tokenizer_vocab': tokenizer.vocab
        }, final_model_path)
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"📁 Results saved to: {args.output_dir}")
        logger.info(f"🔢 Total training errors: {trainer.error_count}")
        logger.info(f"📊 Total batches processed: {trainer.batch_count}")
        
        # Display performance statistics
        if hasattr(model, 'get_performance_stats'):
            try:
                perf_stats = model.get_performance_stats()
                logger.info(f"📈 Performance Stats: {perf_stats}")
            except:
                pass
        
        # MPS statistics if available
        if args.use_mps and MPS_AVAILABLE:
            for name, module in model.named_modules():
                if hasattr(module, 'get_mps_stats'):
                    try:
                        mps_stats = module.get_mps_stats()
                        logger.info(f"🔧 MPS Stats for {name}: {mps_stats}")
                    except:
                        pass
        
        # Generate sample text
        logger.info("Generating sample text...")
        model.eval()
        try:
            with torch.no_grad():
                # Generate from a simple prompt
                prompt = "The future of artificial intelligence"
                prompt_tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([prompt_tokens[:min(len(prompt_tokens), args.seq_length//2)]], device=device)
                
                for _ in range(50):  # Generate 50 tokens
                    outputs = model(input_ids)
                    next_token_logits = outputs[0, -1, :]
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
                    
                    if input_ids.size(1) >= args.seq_length:
                        break
                
                generated_tokens = input_ids[0].cpu().tolist()
                generated_text = tokenizer.decode(generated_tokens)
                logger.info(f"🤖 Generated text: {generated_text}")
        
        except Exception as e:
            log_exception(e, "text generation")
            logger.warning("Could not generate sample text")
    
    except Exception as e:
        error_msg = log_exception(e, "enhanced training main")
        logger.critical(f"💥 Training failed: {error_msg}")
        sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced SpectralLLM Training")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to text dataset file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of sample texts to generate if no dataset provided")
    parser.add_argument("--vocab_size", type=int, default=1000,
                       help="Maximum vocabulary size")
    parser.add_argument("--seq_length", type=int, default=128,
                       help="Sequence length for training")
    
    # Model arguments
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
    
    # HRFEvo arguments
    parser.add_argument("--enable_evolution", action="store_true",
                       help="Enable HRFEvo during training")
    parser.add_argument("--evolution_population", type=int, default=10,
                       help="Evolution population size")
    parser.add_argument("--evolution_generations", type=int, default=5,
                       help="Number of evolution generations")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of data loader workers")
    
    # Optimized LR arguments (from train_optimized_lr.py)
    parser.add_argument("--use_optimized_lr", action="store_true", default=True,
                       help="Use optimized learning rate scheduling with cosine annealing")
    parser.add_argument("--target_perplexity", type=float, default=None,
                       help="Target perplexity for early stopping (optional)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                       help="Minimum LR as ratio of peak (maintains 10%% of peak LR)")
    
    # Device and optimization
    parser.add_argument("--use_mps", action="store_true",
                       help="Use MPS optimizations if available")
    
    # Logging and output
    parser.add_argument("--output_dir", type=str, default="./enhanced_training_output",
                       help="Output directory for results")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_enhanced_spectralllm(args) 