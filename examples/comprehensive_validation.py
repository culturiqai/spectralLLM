#!/usr/bin/env python3
"""
Comprehensive SpectralLLM Validation Script
==========================================

Complete validation suite incorporating features from verify_perplexity.py:
- Perplexity calculation verification
- Dataset integrity validation
- Baseline model comparisons
- Training step verification
- Comprehensive reporting
"""

import os
import sys
import argparse
import logging
import json
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

# SpectralLLM imports
import spectralllm
from spectralllm.training.trainer import SpectralTrainer, EnhancedTextDataset
from spectralllm.core.config import Config
from spectralllm.validation import PerplexityValidator, DatasetValidator, BaselineEvaluator


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file or "comprehensive_validation.log")
        ]
    )
    return logging.getLogger(__name__)


def create_sample_datasets(args):
    """Create sample datasets for validation"""
    texts = []
    
    # Create diverse sample data patterns
    patterns = [
        "Deep learning revolutionizes artificial intelligence through neural networks. ",
        "Machine learning algorithms learn complex patterns from large datasets. ",
        "Natural language processing enables computers to understand human language. ",
        "Transformer architectures are the foundation of modern language models. ",
        "Wavelet analysis provides multi-resolution decomposition of signals. ",
        "Spectral methods offer efficient computation for sequence modeling. ",
        "Optimization algorithms adapt learning rates during neural training. ",
        "Mathematics forms the theoretical foundation of machine learning. ",
        "Gradient descent iteratively minimizes loss functions in networks. ",
        "Attention mechanisms allow models to focus on relevant input parts. ",
    ]
    
    # Generate text samples of varying lengths
    for i in range(args.num_samples):
        sample_length = torch.randint(100, 300, (1,)).item()
        sample_text = ""
        
        while len(sample_text.split()) < sample_length:
            pattern = patterns[i % len(patterns)]
            sample_text += pattern
        
        texts.append(sample_text)
    
    return texts


def run_comprehensive_validation(args):
    """Run complete validation suite"""
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info("🚀 COMPREHENSIVE SPECTRALLLM VALIDATION")
    logger.info("=" * 60)
    
    try:
        # Create output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        validation_dir = os.path.join(args.output_dir, f"validation_{timestamp}")
        os.makedirs(validation_dir, exist_ok=True)
        
        validation_results = {
            'timestamp': timestamp,
            'validation_config': vars(args),
            'results': {}
        }
        
        # 1. PERPLEXITY CALCULATION VALIDATION
        logger.info("1️⃣ VALIDATING PERPLEXITY CALCULATIONS")
        logger.info("-" * 40)
        
        perplexity_validator = PerplexityValidator(logger)
        ppl_validation_passed = perplexity_validator.test_perplexity_calculation()
        
        validation_results['results']['perplexity_validation'] = {
            'passed': ppl_validation_passed,
            'test_completed': True
        }
        
        if not ppl_validation_passed:
            logger.error("❌ Perplexity validation failed - critical issue!")
            return False
        
        # 2. DATASET CREATION AND VALIDATION
        logger.info("\n2️⃣ DATASET CREATION AND VALIDATION")
        logger.info("-" * 40)
        
        # Create or load dataset
        if args.dataset_path and os.path.exists(args.dataset_path):
            logger.info(f"Loading dataset from {args.dataset_path}")
            with open(args.dataset_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
        else:
            logger.info("Creating sample dataset")
            texts = create_sample_datasets(args)
        
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
        
        # Create enhanced datasets with validation
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
        
        # Validate datasets
        dataset_validator = DatasetValidator(logger)
        
        # Check for data leakage
        overlap_results = dataset_validator.check_dataset_overlap(
            train_dataset, val_dataset, max_samples=args.overlap_check_samples
        )
        
        # Verify sequence construction
        sequence_results = dataset_validator.verify_sequence_construction(
            train_dataset, expected_stride=args.seq_length, expected_seq_length=args.seq_length
        )
        
        # Validate training steps
        step_results = dataset_validator.validate_training_steps(
            train_dataset, args.batch_size
        )
        
        validation_results['results']['dataset_validation'] = {
            'overlap_check': overlap_results,
            'sequence_construction': sequence_results,
            'training_steps': step_results,
            'train_dataset_stats': train_dataset.get_dataset_stats(),
            'val_dataset_stats': val_dataset.get_dataset_stats()
        }
        
        # 3. MODEL CREATION AND CONFIGURATION
        logger.info("\n3️⃣ MODEL CREATION AND CONFIGURATION")
        logger.info("-" * 40)
        
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
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Create SpectralLLM model
        logger.info("Creating SpectralLLM model...")
        model = spectralllm.SpectralLLM(config)
        spectral_params = model.count_parameters()
        logger.info(f"SpectralLLM parameters: {spectral_params:,}")
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f"Using device: {device}")
        
        # Test forward pass
        logger.info("Testing SpectralLLM forward pass...")
        model.eval()
        with torch.no_grad():
            test_input = torch.randint(0, vocab_size, (2, args.seq_length), device=device)
            test_output = model(test_input)
            logger.info(f"✅ SpectralLLM forward pass successful: {test_output.shape}")
        
        validation_results['results']['model_validation'] = {
            'config': config.to_dict(),
            'parameters': spectral_params,
            'forward_pass_successful': True,
            'device': str(device)
        }
        
        # 4. BASELINE EVALUATION
        logger.info("\n4️⃣ BASELINE MODEL EVALUATION")
        logger.info("-" * 40)
        
        # Create data loaders for evaluation
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Evaluate baseline models
        baseline_evaluator = BaselineEvaluator(device=str(device), logger=logger)
        baseline_results = baseline_evaluator.evaluate_multiple_baselines(
            val_loader, vocab_size, args.seq_length
        )
        
        validation_results['results']['baseline_evaluation'] = baseline_results
        
        # 5. SPECTRALLLM EVALUATION
        logger.info("\n5️⃣ SPECTRALLLM EVALUATION")
        logger.info("-" * 40)
        
        # Create trainer for SpectralLLM evaluation
        trainer = SpectralTrainer(model, config, use_optimized_lr=False)  # Just for evaluation
        
        # Evaluate SpectralLLM (without training - just random weights)
        spectral_metrics = trainer.evaluate(val_loader)
        spectral_metrics['parameters'] = spectral_params
        spectral_metrics['model_name'] = 'SpectralLLM'
        
        logger.info(f"SpectralLLM (untrained) Results:")
        logger.info(f"  Loss: {spectral_metrics['val_loss']:.4f}")
        logger.info(f"  Perplexity: {spectral_metrics['val_perplexity']:.2f}")
        logger.info(f"  Accuracy: {spectral_metrics['val_accuracy']:.4f}")
        
        # 6. COMPARATIVE ANALYSIS
        logger.info("\n6️⃣ COMPARATIVE ANALYSIS")
        logger.info("-" * 40)
        
        comparison_results = baseline_evaluator.compare_with_spectralllm(
            spectral_metrics, baseline_results
        )
        
        validation_results['results']['spectralllm_evaluation'] = spectral_metrics
        validation_results['results']['comparative_analysis'] = comparison_results
        
        # 7. PERPLEXITY INTERPRETATION
        logger.info("\n7️⃣ PERPLEXITY INTERPRETATION")
        logger.info("-" * 40)
        
        # Interpret perplexity values
        interpretations = {}
        
        # SpectralLLM interpretation
        spectral_interpretation = perplexity_validator.get_perplexity_interpretation(
            spectral_metrics['val_perplexity'], vocab_size, f"SpectralLLM ({spectral_params//1000}K params)"
        )
        interpretations['spectralllm'] = spectral_interpretation
        
        logger.info(f"SpectralLLM Interpretation:")
        logger.info(f"  Perplexity: {spectral_interpretation['perplexity']}")
        logger.info(f"  Quality: {spectral_interpretation['quality']}")
        logger.info(f"  Context: {spectral_interpretation['context']}")
        
        # Baseline interpretations
        for name, metrics in baseline_results.items():
            interp = perplexity_validator.get_perplexity_interpretation(
                metrics['perplexity'], vocab_size, f"{metrics['model_name']} ({metrics['parameters']//1000}K params)"
            )
            interpretations[name] = interp
            logger.info(f"{metrics['model_name']} Quality: {interp['quality']}")
        
        validation_results['results']['perplexity_interpretations'] = interpretations
        
        # 8. VALIDATION SUMMARY
        logger.info("\n8️⃣ VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        # Create summary
        summary = {
            'total_validations': 0,
            'passed_validations': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check perplexity validation
        summary['total_validations'] += 1
        if ppl_validation_passed:
            summary['passed_validations'] += 1
            logger.info("✅ Perplexity calculation: PASSED")
        else:
            summary['critical_issues'].append("Perplexity calculation failed")
            logger.error("❌ Perplexity calculation: FAILED")
        
        # Check dataset overlap
        summary['total_validations'] += 1
        if overlap_results['assessment'] == 'low':
            summary['passed_validations'] += 1
            logger.info("✅ Dataset overlap: ACCEPTABLE")
        elif overlap_results['assessment'] == 'medium':
            summary['warnings'].append(f"Medium dataset overlap: {overlap_results['overlap_percentage']:.2%}")
            logger.warning(f"⚠️ Dataset overlap: MEDIUM ({overlap_results['overlap_percentage']:.2%})")
        else:
            summary['critical_issues'].append(f"High dataset overlap: {overlap_results['overlap_percentage']:.2%}")
            logger.error(f"❌ Dataset overlap: HIGH ({overlap_results['overlap_percentage']:.2%})")
        
        # Check sequence construction
        summary['total_validations'] += 1
        if sequence_results['construction_quality'] == 'excellent':
            summary['passed_validations'] += 1
            logger.info("✅ Sequence construction: EXCELLENT")
        else:
            summary['warnings'].append(f"Sequence construction: {sequence_results['construction_quality']}")
            logger.warning(f"⚠️ Sequence construction: {sequence_results['construction_quality'].upper()}")
        
        # Check SpectralLLM vs baselines
        summary['total_validations'] += 1
        if comparison_results['analysis']['overall_assessment']['better_than_best_baseline']:
            summary['passed_validations'] += 1
            logger.info("✅ SpectralLLM performance: BETTER THAN BASELINES")
        else:
            tier = comparison_results['analysis']['overall_assessment']['performance_tier']
            if tier in ['above_median', 'below_median']:
                summary['warnings'].append(f"SpectralLLM performance tier: {tier}")
                logger.warning(f"⚠️ SpectralLLM performance: {tier.upper().replace('_', ' ')}")
            else:
                summary['critical_issues'].append(f"SpectralLLM performance: {tier}")
                logger.error(f"❌ SpectralLLM performance: {tier.upper()}")
        
        # Generate recommendations
        if overlap_results['assessment'] != 'low':
            summary['recommendations'].append("Consider reviewing dataset construction to reduce overlap")
        
        if spectral_metrics['val_perplexity'] > vocab_size * 0.1:
            summary['recommendations'].append("SpectralLLM perplexity is high - consider training or architectural improvements")
        
        if not comparison_results['analysis']['overall_assessment']['better_than_best_baseline']:
            summary['recommendations'].append("SpectralLLM not outperforming baselines - consider model enhancements")
        
        validation_results['results']['validation_summary'] = summary
        
        # Final validation score
        validation_score = summary['passed_validations'] / summary['total_validations'] * 100
        logger.info(f"\n🏆 OVERALL VALIDATION SCORE: {validation_score:.1f}% ({summary['passed_validations']}/{summary['total_validations']})")
        
        if validation_score >= 75:
            logger.info("✅ VALIDATION: PASSED")
        elif validation_score >= 50:
            logger.warning("⚠️ VALIDATION: PASSED WITH WARNINGS")
        else:
            logger.error("❌ VALIDATION: FAILED")
        
        # Save complete results
        results_file = os.path.join(validation_dir, "comprehensive_validation_results.json")
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"\n📁 Complete validation results saved to: {results_file}")
        
        # Save individual component results
        dataset_validator.save_validation_results(
            validation_results['results']['dataset_validation'], 
            validation_dir, timestamp
        )
        
        baseline_evaluator.save_baseline_results(
            validation_results['results']['baseline_evaluation'], 
            validation_dir, timestamp
        )
        
        return validation_score >= 50
        
    except Exception as e:
        logger.critical(f"💥 Validation failed with error: {str(e)}")
        import traceback
        logger.critical(f"Traceback: {traceback.format_exc()}")
        return False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Comprehensive SpectralLLM Validation")
    
    # Data parameters
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to text dataset file")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="Number of sample texts if no dataset provided")
    parser.add_argument("--vocab_size", type=int, default=1000,
                       help="Maximum vocabulary size")
    parser.add_argument("--seq_length", type=int, default=256,
                       help="Sequence length for validation")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    
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
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    
    # Validation parameters
    parser.add_argument("--overlap_check_samples", type=int, default=100,
                       help="Number of samples to check for overlap")
    
    # Output and logging
    parser.add_argument("--output_dir", type=str, default="./validation_results",
                       help="Output directory for validation results")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("🚀 COMPREHENSIVE SPECTRALLLM VALIDATION SUITE")
    print("Integrating insights from verify_perplexity.py")
    print("=" * 70)
    
    success = run_comprehensive_validation(args)
    
    if success:
        print("\n✅ VALIDATION COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED")
        sys.exit(1) 