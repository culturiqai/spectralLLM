#!/usr/bin/env python3
"""
Metrics and Data Leakage Validation for SpectralLLM
==================================================

Comprehensive validation of training metrics, data integrity,
and leakage detection based on the reference implementation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import hashlib
import json
import os
from collections import defaultdict
import warnings
import logging
from tqdm import tqdm

# Try to import SpectralLLM components
try:
    from ..utils.performance import PerformanceProfiler, get_profiler
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

# Try to import transformers for baseline evaluation
try:
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    from torch.utils.data import DataLoader
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class MetricsValidator:
    """
    Comprehensive metrics validation and data leakage detection.
    
    Features:
    - Training/validation metrics consistency checking
    - Data leakage detection between train/val/test sets
    - Sequence overlap analysis
    - Statistical validation of training progress
    - Perplexity calculation verification
    - Baseline model evaluation
    - Non-overlapping sequence validation
    - Comprehensive reporting
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize metrics validator.
        
        Args:
            tolerance: Numerical tolerance for floating point comparisons
        """
        self.tolerance = tolerance
        self.results = {}
        self.data_fingerprints = {}
        
        # Initialize profiler if available
        self.profiler = get_profiler() if PERFORMANCE_AVAILABLE else None
    
    def validate_training_metrics(self, training_history: Dict[str, List[float]], 
                                verbose: bool = True) -> Dict[str, Any]:
        """
        Validate training metrics for consistency and expected behavior.
        
        Args:
            training_history: Dictionary with training metrics over time
            verbose: Whether to print detailed validation results
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'metrics_consistency': True,
            'issues_found': [],
            'recommendations': [],
            'statistics': {}
        }
        
        # Check for required metrics
        required_metrics = ['train_loss', 'val_loss']
        for metric in required_metrics:
            if metric not in training_history:
                validation_results['metrics_consistency'] = False
                validation_results['issues_found'].append(f"Missing required metric: {metric}")
        
        # Validate individual metrics if they exist
        for metric_name, values in training_history.items():
            if not values:
                continue
                
            metric_stats = self._analyze_metric_sequence(metric_name, values)
            validation_results['statistics'][metric_name] = metric_stats
            
            # Check for anomalies
            issues = self._detect_metric_anomalies(metric_name, values, metric_stats)
            validation_results['issues_found'].extend(issues)
        
        # Cross-metric validation
        cross_issues = self._validate_metric_relationships(training_history)
        validation_results['issues_found'].extend(cross_issues)
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(
            validation_results['issues_found'], 
            validation_results['statistics']
        )
        
        if validation_results['issues_found']:
            validation_results['metrics_consistency'] = False
        
        if verbose:
            self._print_validation_results(validation_results)
        
        self.results['metrics_validation'] = validation_results
        return validation_results
    
    def _analyze_metric_sequence(self, metric_name: str, values: List[float]) -> Dict[str, Any]:
        """Analyze a sequence of metric values for statistical properties"""
        values_array = np.array(values)
        
        stats = {
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'first': values[0],
            'last': values[-1],
            'range': float(np.max(values_array) - np.min(values_array))
        }
        
        # Calculate trends
        if len(values) > 1:
            # Simple linear trend
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values_array, 1)
            stats['trend_slope'] = float(slope)
            stats['trend_direction'] = 'decreasing' if slope < -self.tolerance else ('increasing' if slope > self.tolerance else 'stable')
            
            # Improvement rate (for loss metrics)
            if 'loss' in metric_name.lower():
                improvement = (values[0] - values[-1]) / values[0] if values[0] != 0 else 0
                stats['improvement_rate'] = float(improvement)
            
            # Volatility (coefficient of variation)
            if stats['mean'] != 0:
                stats['volatility'] = stats['std'] / abs(stats['mean'])
            else:
                stats['volatility'] = float('inf')
        
        return stats
    
    def _detect_metric_anomalies(self, metric_name: str, values: List[float], 
                                stats: Dict[str, Any]) -> List[str]:
        """Detect anomalies in metric sequences"""
        issues = []
        
        # Check for NaN or infinite values
        for i, value in enumerate(values):
            if not np.isfinite(value):
                issues.append(f"{metric_name}: Non-finite value {value} at step {i}")
        
        # Check for unexpected trends
        if 'loss' in metric_name.lower():
            # Loss should generally decrease over time
            if stats.get('trend_direction') == 'increasing':
                issues.append(f"{metric_name}: Loss is increasing over time (slope: {stats.get('trend_slope', 0):.6f})")
            
            # Check for training vs validation loss divergence
            if 'train' in metric_name.lower() and stats.get('improvement_rate', 0) > 0.1:
                # If training loss improved significantly, check if validation follows
                pass  # This requires cross-metric analysis
        
        # Check for excessive volatility
        if stats.get('volatility', 0) > 2.0:
            issues.append(f"{metric_name}: High volatility detected (CV: {stats['volatility']:.2f})")
        
        # Check for stagnation
        if len(values) > 10:
            recent_values = values[-10:]
            recent_std = np.std(recent_values)
            if recent_std < self.tolerance:
                issues.append(f"{metric_name}: Possible stagnation in last 10 steps (std: {recent_std:.8f})")
        
        return issues
    
    def _validate_metric_relationships(self, training_history: Dict[str, List[float]]) -> List[str]:
        """Validate relationships between different metrics"""
        issues = []
        
        # Check train vs validation loss relationship
        if 'train_loss' in training_history and 'val_loss' in training_history:
            train_loss = training_history['train_loss']
            val_loss = training_history['val_loss']
            
            if len(train_loss) == len(val_loss) and len(train_loss) > 0:
                # Check for overfitting (validation loss increasing while training decreases)
                min_length = min(len(train_loss), len(val_loss))
                train_slice = train_loss[:min_length]
                val_slice = val_loss[:min_length]
                
                # Calculate correlation
                if len(train_slice) > 1:
                    correlation = np.corrcoef(train_slice, val_slice)[0, 1]
                    if not np.isnan(correlation) and correlation < 0.5:
                        issues.append(f"Low correlation between train and validation loss: {correlation:.3f}")
                
                # Check for overfitting pattern
                if len(train_slice) > 5:
                    recent_train_trend = np.polyfit(range(5), train_slice[-5:], 1)[0]
                    recent_val_trend = np.polyfit(range(5), val_slice[-5:], 1)[0]
                    
                    if recent_train_trend < -self.tolerance and recent_val_trend > self.tolerance:
                        issues.append("Potential overfitting: training loss decreasing while validation loss increasing")
        
        # Check accuracy vs loss relationship
        if 'train_loss' in training_history and 'train_accuracy' in training_history:
            train_loss = training_history['train_loss']
            train_acc = training_history['train_accuracy']
            
            min_length = min(len(train_loss), len(train_acc))
            if min_length > 1:
                loss_slice = train_loss[:min_length]
                acc_slice = train_acc[:min_length]
                
                # They should be negatively correlated
                correlation = np.corrcoef(loss_slice, acc_slice)[0, 1]
                if not np.isnan(correlation) and correlation > -0.3:
                    issues.append(f"Unexpected loss-accuracy correlation: {correlation:.3f} (should be negative)")
        
        return issues
    
    def _generate_recommendations(self, issues: List[str], 
                                statistics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on detected issues"""
        recommendations = []
        
        # Analyze issues and provide recommendations
        issue_text = ' '.join(issues).lower()
        
        if 'overfitting' in issue_text:
            recommendations.append("Consider regularization techniques (dropout, weight decay)")
            recommendations.append("Implement early stopping based on validation metrics")
            recommendations.append("Reduce model complexity or increase training data")
        
        if 'stagnation' in issue_text:
            recommendations.append("Consider learning rate adjustment or scheduler changes")
            recommendations.append("Check for gradient clipping issues")
            recommendations.append("Verify data shuffling and augmentation")
        
        if 'volatility' in issue_text or 'high' in issue_text:
            recommendations.append("Consider reducing learning rate")
            recommendations.append("Implement gradient clipping")
            recommendations.append("Use batch normalization or layer normalization")
        
        if 'increasing' in issue_text and 'loss' in issue_text:
            recommendations.append("Check for learning rate that's too high")
            recommendations.append("Verify gradient computation and backpropagation")
            recommendations.append("Consider warm-up for learning rate scheduling")
        
        return recommendations
    
    def detect_data_leakage(self, train_data: List[str], val_data: List[str], 
                          test_data: Optional[List[str]] = None,
                          sequence_overlap_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Detect data leakage between training, validation, and test sets.
        
        Args:
            train_data: Training data sequences
            val_data: Validation data sequences  
            test_data: Optional test data sequences
            sequence_overlap_threshold: Threshold for considering sequences as overlapping
            
        Returns:
            Leakage detection results
        """
        if self.profiler:
            with self.profiler.profile_operation("data_leakage_detection"):
                return self._detect_data_leakage_impl(
                    train_data, val_data, test_data, sequence_overlap_threshold
                )
        else:
            return self._detect_data_leakage_impl(
                train_data, val_data, test_data, sequence_overlap_threshold
            )
    
    def _detect_data_leakage_impl(self, train_data: List[str], val_data: List[str], 
                                test_data: Optional[List[str]], 
                                sequence_overlap_threshold: float) -> Dict[str, Any]:
        """Implementation of data leakage detection"""
        results = {
            'exact_duplicates': {},
            'sequence_overlaps': {},
            'hash_collisions': {},
            'leakage_detected': False,
            'recommendations': []
        }
        
        # Create fingerprints for each dataset
        datasets = {'train': train_data, 'val': val_data}
        if test_data is not None:
            datasets['test'] = test_data
        
        fingerprints = {}
        for name, data in datasets.items():
            fingerprints[name] = self._create_data_fingerprints(data)
        
        # Check for exact duplicates
        for name1, fp1 in fingerprints.items():
            for name2, fp2 in fingerprints.items():
                if name1 >= name2:  # Avoid duplicate comparisons
                    continue
                
                exact_matches = set(fp1['hashes']) & set(fp2['hashes'])
                if exact_matches:
                    results['exact_duplicates'][f"{name1}_vs_{name2}"] = {
                        'count': len(exact_matches),
                        'percentage': len(exact_matches) / min(len(fp1['hashes']), len(fp2['hashes'])) * 100
                    }
                    results['leakage_detected'] = True
        
        # Check for sequence overlaps
        for name1, data1 in datasets.items():
            for name2, data2 in datasets.items():
                if name1 >= name2:
                    continue
                
                overlaps = self._find_sequence_overlaps(
                    data1, data2, sequence_overlap_threshold
                )
                if overlaps:
                    results['sequence_overlaps'][f"{name1}_vs_{name2}"] = {
                        'count': len(overlaps),
                        'examples': overlaps[:5],  # First 5 examples
                        'max_overlap': max(overlap['overlap_ratio'] for overlap in overlaps)
                    }
                    if max(overlap['overlap_ratio'] for overlap in overlaps) > sequence_overlap_threshold:
                        results['leakage_detected'] = True
        
        # Generate recommendations
        if results['leakage_detected']:
            results['recommendations'].extend([
                "Remove duplicate sequences between datasets",
                "Implement proper data splitting with temporal or semantic boundaries",
                "Consider using stratified sampling for balanced splits",
                "Validate data preprocessing pipeline for leakage sources"
            ])
        
        if results['exact_duplicates']:
            results['recommendations'].append(
                "Exact duplicates found - implement deduplication before splitting"
            )
        
        if results['sequence_overlaps']:
            results['recommendations'].append(
                "High sequence overlap detected - review splitting methodology"
            )
        
        self.results['data_leakage'] = results
        return results
    
    def _create_data_fingerprints(self, data: List[str]) -> Dict[str, Any]:
        """Create fingerprints for data sequences"""
        fingerprints = {
            'hashes': [],
            'lengths': [],
            'char_counts': []
        }
        
        for sequence in data:
            # Create hash
            hash_obj = hashlib.sha256(sequence.encode('utf-8'))
            fingerprints['hashes'].append(hash_obj.hexdigest())
            
            # Store length
            fingerprints['lengths'].append(len(sequence))
            
            # Character frequency
            char_count = defaultdict(int)
            for char in sequence:
                char_count[char] += 1
            fingerprints['char_counts'].append(dict(char_count))
        
        return fingerprints
    
    def _find_sequence_overlaps(self, data1: List[str], data2: List[str], 
                              threshold: float) -> List[Dict[str, Any]]:
        """Find overlapping sequences between two datasets"""
        overlaps = []
        
        for i, seq1 in enumerate(data1[:100]):  # Limit for performance
            for j, seq2 in enumerate(data2[:100]):
                overlap_ratio = self._calculate_sequence_overlap(seq1, seq2)
                if overlap_ratio > threshold:
                    overlaps.append({
                        'data1_idx': i,
                        'data2_idx': j,
                        'overlap_ratio': overlap_ratio,
                        'seq1_preview': seq1[:100],
                        'seq2_preview': seq2[:100]
                    })
        
        return overlaps
    
    def _calculate_sequence_overlap(self, seq1: str, seq2: str) -> float:
        """Calculate overlap ratio between two sequences using longest common subsequence"""
        # Simple implementation - could be optimized for large sequences
        if not seq1 or not seq2:
            return 0.0
        
        # Use a sliding window approach for efficiency
        max_overlap = 0
        window_size = min(len(seq1), len(seq2), 1000)  # Limit window size
        
        # Check multiple positions
        for i in range(0, len(seq1) - window_size + 1, window_size // 2):
            for j in range(0, len(seq2) - window_size + 1, window_size // 2):
                window1 = seq1[i:i + window_size]
                window2 = seq2[j:j + window_size]
                
                # Calculate character-level overlap
                overlap = sum(1 for c1, c2 in zip(window1, window2) if c1 == c2)
                overlap_ratio = overlap / window_size
                max_overlap = max(max_overlap, overlap_ratio)
        
        return max_overlap
    
    def validate_non_overlapping_sequences(self, sequences: List[str], 
                                         stride: int, sequence_length: int,
                                         verbose: bool = True) -> Dict[str, Any]:
        """
        Validate that sequences are properly non-overlapping based on stride and length.
        
        Args:
            sequences: List of sequences to validate
            stride: Expected stride between sequences
            sequence_length: Expected length of each sequence
            verbose: Whether to print validation results
            
        Returns:
            Validation results
        """
        results = {
            'is_non_overlapping': True,
            'issues_found': [],
            'statistics': {},
            'recommendations': []
        }
        
        # Check sequence lengths
        lengths = [len(seq) for seq in sequences]
        length_stats = {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'expected': sequence_length
        }
        results['statistics']['sequence_lengths'] = length_stats
        
        # Check for length consistency
        if length_stats['std'] > self.tolerance:
            results['issues_found'].append(f"Inconsistent sequence lengths (std: {length_stats['std']:.2f})")
            results['is_non_overlapping'] = False
        
        # Check expected vs actual lengths
        if abs(length_stats['mean'] - sequence_length) > self.tolerance:
            results['issues_found'].append(
                f"Sequence length mismatch: expected {sequence_length}, got mean {length_stats['mean']:.2f}"
            )
        
        # Sample-based overlap checking (for performance)
        sample_size = min(100, len(sequences))
        sample_indices = np.random.choice(len(sequences), sample_size, replace=False)
        
        overlap_count = 0
        for i in range(len(sample_indices) - 1):
            idx1, idx2 = sample_indices[i], sample_indices[i + 1]
            if abs(idx1 - idx2) == 1:  # Adjacent sequences
                overlap_ratio = self._calculate_sequence_overlap(sequences[idx1], sequences[idx2])
                expected_overlap = max(0, (sequence_length - stride) / sequence_length)
                
                if abs(overlap_ratio - expected_overlap) > 0.1:  # 10% tolerance
                    overlap_count += 1
        
        if overlap_count > sample_size * 0.1:  # More than 10% of samples have unexpected overlap
            results['issues_found'].append(f"Unexpected overlap patterns in {overlap_count}/{sample_size} samples")
            results['is_non_overlapping'] = False
        
        # Generate recommendations
        if results['issues_found']:
            if 'length' in ' '.join(results['issues_found']).lower():
                results['recommendations'].append("Review sequence tokenization and padding logic")
            if 'overlap' in ' '.join(results['issues_found']).lower():
                results['recommendations'].append("Check stride calculation in dataset creation")
                results['recommendations'].append(f"Verify stride={stride} with sequence_length={sequence_length}")
        
        if verbose:
            print(f"Non-overlapping validation: {'✅ PASSED' if results['is_non_overlapping'] else '❌ FAILED'}")
            if results['issues_found']:
                for issue in results['issues_found']:
                    print(f"  ⚠️  {issue}")
            if results['recommendations']:
                print("Recommendations:")
                for rec in results['recommendations']:
                    print(f"  • {rec}")
        
        self.results['non_overlapping_validation'] = results
        return results
    
    def _print_validation_results(self, results: Dict[str, Any]):
        """Print formatted validation results"""
        print("\n" + "="*60)
        print("METRICS VALIDATION RESULTS")
        print("="*60)
        
        status = "✅ PASSED" if results['metrics_consistency'] else "❌ FAILED"
        print(f"Overall Status: {status}")
        
        if results['issues_found']:
            print(f"\nIssues Found ({len(results['issues_found'])}):")
            for issue in results['issues_found']:
                print(f"  ⚠️  {issue}")
        
        if results['statistics']:
            print(f"\nMetrics Statistics:")
            for metric, stats in results['statistics'].items():
                print(f"  {metric}:")
                print(f"    Range: {stats['min']:.4f} - {stats['max']:.4f}")
                print(f"    Trend: {stats.get('trend_direction', 'unknown')}")
                if 'improvement_rate' in stats:
                    print(f"    Improvement: {stats['improvement_rate']*100:.1f}%")
        
        if results['recommendations']:
            print(f"\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  • {rec}")
        
        print("="*60)
    
    def generate_comprehensive_report(self, output_path: str):
        """Generate comprehensive validation report"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"📊 Comprehensive validation report saved: {output_path}")
        except Exception as e:
            print(f"⚠️  Failed to save validation report: {e}")
    
    def reset_results(self):
        """Reset validation results"""
        self.results = {}
        self.data_fingerprints = {}
    
    def test_perplexity_calculation(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive test of perplexity calculation correctness.
        
        Tests multiple scenarios:
        - Perfect predictions (PPL should be 1.0)
        - Random predictions (PPL should be ~vocab_size)
        - Multi-token averaging
        - Weighted loss calculations
        
        Args:
            verbose: Whether to print detailed test results
            
        Returns:
            Test results dictionary
        """
        if verbose:
            logger.info("Testing perplexity calculation with known values...")
        
        test_results = {
            'all_tests_passed': True,
            'test_cases': {},
            'errors': []
        }
        
        try:
            # Test Case 1: Perfect prediction (loss=0, PPL=1)
            perfect_loss = torch.tensor(0.0)
            perfect_ppl = torch.exp(perfect_loss)
            
            test_results['test_cases']['perfect_prediction'] = {
                'loss': perfect_loss.item(),
                'perplexity': perfect_ppl.item(),
                'expected_ppl': 1.0,
                'passed': abs(perfect_ppl.item() - 1.0) < self.tolerance
            }
            
            if verbose:
                logger.info(f"Perfect prediction - Loss: {perfect_loss:.4f}, PPL: {perfect_ppl:.4f}")
            
            # Test Case 2: Random prediction with known vocab size
            vocab_size = 10000
            random_loss = torch.tensor(np.log(vocab_size))
            random_ppl = torch.exp(random_loss)
            
            test_results['test_cases']['random_prediction'] = {
                'loss': random_loss.item(),
                'perplexity': random_ppl.item(),
                'expected_ppl': vocab_size,
                'passed': abs(random_ppl.item() - vocab_size) < 10
            }
            
            if verbose:
                logger.info(f"Random prediction (10K vocab) - Loss: {random_loss:.4f}, PPL: {random_ppl:.2f}")
            
            # Test Case 3: Multi-token average
            multi_token_losses = torch.tensor([1.0, 2.0, 3.0])
            avg_loss = multi_token_losses.mean()
            multi_token_ppl = torch.exp(avg_loss)
            expected_ppl = torch.exp(torch.tensor(2.0))
            
            test_results['test_cases']['multi_token_average'] = {
                'losses': multi_token_losses.tolist(),
                'avg_loss': avg_loss.item(),
                'perplexity': multi_token_ppl.item(),
                'expected_ppl': expected_ppl.item(),
                'passed': abs(multi_token_ppl.item() - expected_ppl.item()) < 0.001
            }
            
            if verbose:
                logger.info(f"Multi-token average - Loss: {avg_loss:.4f}, PPL: {multi_token_ppl:.4f}")
            
            # Test Case 4: Weighted loss with varying token counts
            batch_losses = [
                {"loss": 2.0, "tokens": 100},
                {"loss": 3.0, "tokens": 50},
                {"loss": 1.0, "tokens": 150}
            ]
            
            total_loss = sum(batch["loss"] * batch["tokens"] for batch in batch_losses)
            total_tokens = sum(batch["tokens"] for batch in batch_losses)
            weighted_avg_loss = total_loss / total_tokens
            weighted_ppl = torch.exp(torch.tensor(weighted_avg_loss))
            expected_weighted_loss = (2.0*100 + 3.0*50 + 1.0*150) / 300
            
            test_results['test_cases']['weighted_average'] = {
                'batch_data': batch_losses,
                'weighted_avg_loss': weighted_avg_loss,
                'perplexity': weighted_ppl.item(),
                'expected_loss': expected_weighted_loss,
                'passed': abs(weighted_avg_loss - expected_weighted_loss) < 0.001
            }
            
            if verbose:
                logger.info(f"Weighted average - Loss: {weighted_avg_loss:.4f}, PPL: {weighted_ppl:.4f}")
            
            # Test Case 5: Realistic language model scenario
            batch_size, seq_length, vocab_size = 4, 10, 100
            targets = torch.randint(0, vocab_size, (batch_size, seq_length))
            
            # Create realistic logits where correct token has higher probability
            realistic_logits = torch.randn(batch_size, seq_length, vocab_size) * 0.1
            for b in range(batch_size):
                for s in range(seq_length):
                    correct_idx = targets[b, s].item()
                    realistic_logits[b, s, correct_idx] = 2.0  # Boost correct token
                    
                    # Boost a few alternatives
                    for k in range(5):
                        alt_idx = (correct_idx + k + 1) % vocab_size
                        realistic_logits[b, s, alt_idx] = 1.0
            
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            loss_per_token = criterion(realistic_logits.reshape(-1, vocab_size), targets.reshape(-1))
            
            # Correct method: average loss then exp
            avg_loss = loss_per_token.mean().item()
            correct_ppl = torch.exp(torch.tensor(avg_loss)).item()
            
            # Incorrect method: exp per token then average
            incorrect_ppl = torch.exp(loss_per_token).mean().item()
            
            test_results['test_cases']['realistic_scenario'] = {
                'correct_ppl': correct_ppl,
                'incorrect_ppl': incorrect_ppl,
                'difference': abs(correct_ppl - incorrect_ppl),
                'passed': correct_ppl != incorrect_ppl  # They should be different
            }
            
            if verbose:
                logger.info(f"Realistic predictions - Correct PPL: {correct_ppl:.4f}, Incorrect PPL: {incorrect_ppl:.4f}")
            
            # Check if all tests passed
            all_passed = all(test['passed'] for test in test_results['test_cases'].values())
            test_results['all_tests_passed'] = all_passed
            
            if verbose:
                if all_passed:
                    logger.info("✅ Perplexity calculation verification PASSED")
                else:
                    logger.error("❌ Perplexity calculation verification FAILED")
            
        except Exception as e:
            test_results['all_tests_passed'] = False
            test_results['errors'].append(str(e))
            if verbose:
                logger.error(f"Error during perplexity testing: {e}")
        
        self.results['perplexity_validation'] = test_results
        return test_results
    
    def evaluate_gpt2_baseline(self, dataset_loader: Optional[Any] = None, 
                              batch_size: int = 1, seq_length: int = 512, 
                              device: str = 'cpu', model_config: Optional[Any] = None,
                              verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate a baseline GPT-2 model for comparison.
        
        Args:
            dataset_loader: DataLoader for evaluation dataset
            batch_size: Batch size for evaluation
            seq_length: Sequence length
            device: Device to run evaluation on
            model_config: Optional GPT-2 configuration
            verbose: Whether to print detailed results
            
        Returns:
            Baseline evaluation results
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, skipping baseline evaluation")
            return {'error': 'transformers_not_available'}
        
        if verbose:
            logger.info("Evaluating GPT-2 baseline model...")
        
        try:
            # Create model configuration if not provided
            if not model_config:
                if TRANSFORMERS_AVAILABLE:
                    model_config = GPT2Config(
                        vocab_size=50257,
                        n_positions=1024,
                        n_embd=256,    # Small embedding dimension
                        n_layer=4,     # Fewer layers
                        n_head=8,      # Standard head count
                    )
                else:
                    return {'error': 'no_model_config_and_no_transformers'}
            
            model = GPT2LMHeadModel(model_config)
            model.to(device)
            model.eval()
            
            num_params = sum(p.numel() for p in model.parameters())
            
            if verbose:
                logger.info(f"Model parameters: {num_params:,}")
            
            # If no dataset loader provided, create a simple test
            if dataset_loader is None:
                if verbose:
                    logger.info("No dataset provided, creating synthetic test data")
                
                # Create synthetic data for testing
                vocab_size = model_config.vocab_size
                test_inputs = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
                test_targets = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
                
                with torch.no_grad():
                    outputs = model(test_inputs, labels=test_targets)
                    loss = outputs.loss.item()
                    perplexity = torch.exp(torch.tensor(loss)).item()
                
                results = {
                    'model_params': num_params,
                    'test_loss': loss,
                    'test_perplexity': perplexity,
                    'synthetic_data': True
                }
            else:
                # Evaluate on provided dataset
                val_loss = 0.0
                val_tokens = 0
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(tqdm(dataset_loader, desc="Evaluating") if verbose else dataset_loader):
                        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            inputs, targets = batch[0], batch[1]
                        else:
                            inputs = targets = batch
                        
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        outputs = model(inputs, labels=targets)
                        loss = outputs.loss
                        
                        batch_tokens = targets.numel()
                        val_loss += loss.item() * batch_tokens
                        val_tokens += batch_tokens
                
                avg_loss = val_loss / val_tokens
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                
                results = {
                    'model_params': num_params,
                    'val_loss': avg_loss,
                    'val_perplexity': perplexity,
                    'total_tokens': val_tokens,
                    'synthetic_data': False
                }
            
            if verbose:
                logger.info(f"GPT-2 Baseline - Loss: {results.get('val_loss', results.get('test_loss')):.4f}, "
                           f"PPL: {results.get('val_perplexity', results.get('test_perplexity')):.2f}")
            
            self.results['baseline_evaluation'] = results
            return results
            
        except Exception as e:
            error_result = {'error': str(e)}
            if verbose:
                logger.error(f"Error during baseline evaluation: {e}")
            self.results['baseline_evaluation'] = error_result
            return error_result
    
    def check_dataset_overlap(self, train_data: List[str], val_data: List[str], 
                             test_data: Optional[List[str]] = None,
                             seq_length: int = 1024, sample_size: int = 500,
                             verbose: bool = True) -> Dict[str, Any]:
        """
        Enhanced dataset overlap detection with sophisticated analysis.
        
        Args:
            train_data: Training dataset sequences
            val_data: Validation dataset sequences  
            test_data: Optional test dataset sequences
            seq_length: Sequence length for analysis
            sample_size: Number of samples to check for overlap
            verbose: Whether to print detailed results
            
        Returns:
            Overlap analysis results
        """
        if verbose:
            logger.info("Checking for enhanced dataset overlap...")
        
        overlap_results = {
            'train_val_overlap': {},
            'train_test_overlap': {},
            'val_test_overlap': {},
            'high_overlap_samples': [],
            'recommendations': []
        }
        
        try:
            # Check train-validation overlap
            train_val_overlap = self._analyze_sequence_overlap(
                train_data, val_data, "train", "validation", 
                sample_size, verbose
            )
            overlap_results['train_val_overlap'] = train_val_overlap
            
            # Check train-test overlap if test data provided
            if test_data:
                train_test_overlap = self._analyze_sequence_overlap(
                    train_data, test_data, "train", "test", 
                    sample_size, verbose
                )
                overlap_results['train_test_overlap'] = train_test_overlap
                
                # Check validation-test overlap
                val_test_overlap = self._analyze_sequence_overlap(
                    val_data, test_data, "validation", "test", 
                    sample_size, verbose
                )
                overlap_results['val_test_overlap'] = val_test_overlap
            
            # Generate recommendations based on overlap analysis
            overlap_results['recommendations'] = self._generate_overlap_recommendations(overlap_results)
            
            if verbose:
                self._print_overlap_results(overlap_results)
            
            self.results['dataset_overlap'] = overlap_results
            return overlap_results
            
        except Exception as e:
            error_result = {'error': str(e)}
            if verbose:
                logger.error(f"Error during overlap analysis: {e}")
            self.results['dataset_overlap'] = error_result
            return error_result
    
    def _analyze_sequence_overlap(self, data1: List[str], data2: List[str], 
                                 name1: str, name2: str, sample_size: int,
                                 verbose: bool) -> Dict[str, Any]:
        """Analyze overlap between two datasets"""
        sample1_size = min(sample_size, len(data1))
        sample2_size = min(sample_size * 10, len(data2))  # Check against larger sample
        
        if verbose:
            logger.info(f"Checking {sample1_size} {name1} samples against {sample2_size} {name2} samples...")
        
        overlap_count = 0
        high_overlap_samples = []
        exact_matches = 0
        
        # Convert sequences to token lists for analysis
        if isinstance(data1[0], str):
            # If data is text, we need to tokenize or split
            data1_tokens = [seq.split() if isinstance(seq, str) else seq for seq in data1[:sample1_size]]
            data2_tokens = [seq.split() if isinstance(seq, str) else seq for seq in data2[:sample2_size]]
        else:
            # Assume data is already tokenized
            data1_tokens = data1[:sample1_size]
            data2_tokens = data2[:sample2_size]
        
        for i, seq1 in enumerate(tqdm(data1_tokens, desc=f"Checking {name1}-{name2} overlap") if verbose else data1_tokens):
            max_overlap = 0
            best_match_idx = -1
            
            for j, seq2 in enumerate(data2_tokens):
                # Check for exact match first
                if seq1 == seq2:
                    exact_matches += 1
                    max_overlap = 1.0
                    best_match_idx = j
                    break
                
                # Calculate token overlap percentage
                if isinstance(seq1, (list, tuple)) and isinstance(seq2, (list, tuple)):
                    set1, set2 = set(seq1), set(seq2)
                    if len(set1) > 0:
                        overlap_ratio = len(set1.intersection(set2)) / len(set1)
                        if overlap_ratio > max_overlap:
                            max_overlap = overlap_ratio
                            best_match_idx = j
                
                # Check for subsequence matches (sliding window)
                if isinstance(seq1, (list, tuple)) and isinstance(seq2, (list, tuple)):
                    subseq_overlap = self._find_max_subsequence_overlap(seq1, seq2)
                    if subseq_overlap > max_overlap:
                        max_overlap = subseq_overlap
                        best_match_idx = j
            
            if max_overlap > 0.5:  # High overlap threshold
                overlap_count += 1
                high_overlap_samples.append({
                    'sample_id': i,
                    'overlap_ratio': max_overlap,
                    'match_id': best_match_idx,
                    'is_exact': max_overlap == 1.0
                })
        
        return {
            'total_checked': sample1_size,
            'overlap_count': overlap_count,
            'exact_matches': exact_matches,
            'overlap_percentage': overlap_count / sample1_size if sample1_size > 0 else 0,
            'exact_match_percentage': exact_matches / sample1_size if sample1_size > 0 else 0,
            'high_overlap_samples': sorted(high_overlap_samples, key=lambda x: x['overlap_ratio'], reverse=True)[:10]
        }
    
    def _find_max_subsequence_overlap(self, seq1: List, seq2: List, min_length: int = 10) -> float:
        """Find maximum subsequence overlap between two sequences"""
        if len(seq1) < min_length or len(seq2) < min_length:
            return 0.0
        
        max_overlap = 0.0
        
        # Check all subsequences of seq1 of length min_length or more
        for i in range(len(seq1) - min_length + 1):
            for length in range(min_length, min(len(seq1) - i + 1, 50)):  # Limit max length for efficiency
                subseq = seq1[i:i+length]
                
                # Check if this subsequence appears in seq2
                for j in range(len(seq2) - length + 1):
                    if seq2[j:j+length] == subseq:
                        overlap_ratio = length / len(seq1)
                        max_overlap = max(max_overlap, overlap_ratio)
                        break
        
        return max_overlap
    
    def validate_non_overlapping_sequences(self, dataset: Any, seq_length: int, 
                                         expected_stride: Optional[int] = None,
                                         verbose: bool = True) -> Dict[str, Any]:
        """
        Validate that a dataset uses non-overlapping sequences with correct stride.
        
        Args:
            dataset: Dataset object with samples attribute or list of sample indices
            seq_length: Expected sequence length
            expected_stride: Expected stride (defaults to seq_length for non-overlapping)
            verbose: Whether to print detailed results
            
        Returns:
            Validation results
        """
        if expected_stride is None:
            expected_stride = seq_length
        
        if verbose:
            logger.info("Verifying non-overlapping sequences...")
        
        validation_results = {
            'is_non_overlapping': True,
            'stride_correct': True,
            'issues_found': [],
            'statistics': {},
            'recommendations': []
        }
        
        try:
            # Try to get sample indices or positions
            if hasattr(dataset, 'samples'):
                samples = dataset.samples
            elif hasattr(dataset, 'indices'):
                samples = dataset.indices
            elif isinstance(dataset, list):
                samples = dataset
            else:
                validation_results['issues_found'].append("Cannot access dataset sample indices")
                validation_results['is_non_overlapping'] = False
                return validation_results
            
            if len(samples) < 2:
                validation_results['issues_found'].append("Dataset has fewer than 2 samples")
                return validation_results
            
            # Check stride between consecutive samples
            strides = []
            for i in range(len(samples) - 1):
                stride = samples[i + 1] - samples[i]
                strides.append(stride)
            
            # Analyze stride statistics
            unique_strides = list(set(strides))
            most_common_stride = max(set(strides), key=strides.count)
            
            validation_results['statistics'] = {
                'total_samples': len(samples),
                'unique_strides': unique_strides,
                'most_common_stride': most_common_stride,
                'expected_stride': expected_stride,
                'stride_consistency': len(unique_strides) == 1
            }
            
            # Check if stride matches expected value
            if most_common_stride != expected_stride:
                validation_results['stride_correct'] = False
                validation_results['issues_found'].append(
                    f"Incorrect stride! Expected {expected_stride}, got {most_common_stride}"
                )
            
            # Check for overlapping sequences (stride < seq_length)
            if most_common_stride < seq_length:
                validation_results['is_non_overlapping'] = False
                validation_results['issues_found'].append(
                    f"Sequences are overlapping! Stride {most_common_stride} < seq_length {seq_length}"
                )
            
            # Check stride consistency
            if len(unique_strides) > 1:
                validation_results['issues_found'].append(
                    f"Inconsistent stride! Found {len(unique_strides)} different strides: {unique_strides}"
                )
            
            # Calculate expected vs actual sample count
            if hasattr(dataset, 'all_tokens') or hasattr(dataset, 'total_tokens'):
                total_tokens = getattr(dataset, 'all_tokens', getattr(dataset, 'total_tokens', None))
                if hasattr(total_tokens, '__len__'):
                    total_tokens = len(total_tokens)
                
                if total_tokens:
                    expected_samples = (total_tokens - seq_length - 1) // expected_stride
                    actual_samples = len(samples)
                    
                    validation_results['statistics'].update({
                        'total_tokens': total_tokens,
                        'expected_samples': expected_samples,
                        'actual_samples': actual_samples,
                        'sample_efficiency': actual_samples / expected_samples if expected_samples > 0 else 0
                    })
                    
                    # Allow 10% margin for edge cases
                    if abs(expected_samples - actual_samples) > expected_samples * 0.1:
                        validation_results['issues_found'].append(
                            f"Sample count mismatch! Expected ~{expected_samples}, got {actual_samples}"
                        )
            
            # Generate recommendations
            if validation_results['issues_found']:
                validation_results['recommendations'] = [
                    "Ensure dataset uses stride = seq_length for non-overlapping sequences",
                    "Check dataset implementation for correct sample index calculation",
                    "Verify that sequence boundaries are properly handled"
                ]
            
            if verbose:
                self._print_sequence_validation_results(validation_results)
            
            self.results['sequence_validation'] = validation_results
            return validation_results
            
        except Exception as e:
            error_result = {
                'is_non_overlapping': False,
                'stride_correct': False,
                'error': str(e),
                'issues_found': [f"Error during validation: {e}"]
            }
            if verbose:
                logger.error(f"Error during sequence validation: {e}")
            self.results['sequence_validation'] = error_result
            return error_result
    
    def _generate_overlap_recommendations(self, overlap_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on overlap analysis"""
        recommendations = []
        
        # Check train-validation overlap
        train_val = overlap_results.get('train_val_overlap', {})
        if train_val.get('overlap_percentage', 0) > 0.1:  # More than 10% overlap
            recommendations.append(
                f"High train-validation overlap detected ({train_val['overlap_percentage']:.1%}). "
                "Consider using proper data splitting to avoid data leakage."
            )
        
        if train_val.get('exact_match_percentage', 0) > 0.01:  # More than 1% exact matches
            recommendations.append(
                f"Exact sequence matches found between train and validation sets "
                f"({train_val['exact_match_percentage']:.1%}). This indicates data leakage."
            )
        
        # Check other overlaps if available
        for overlap_type in ['train_test_overlap', 'val_test_overlap']:
            overlap_data = overlap_results.get(overlap_type, {})
            if overlap_data.get('overlap_percentage', 0) > 0.05:
                recommendations.append(
                    f"Overlap detected in {overlap_type.replace('_', '-')} "
                    f"({overlap_data['overlap_percentage']:.1%}). Review data splitting strategy."
                )
        
        if not recommendations:
            recommendations.append("No significant data leakage detected. Data splitting appears appropriate.")
        
        return recommendations
    
    def _print_overlap_results(self, results: Dict[str, Any]):
        """Print formatted overlap analysis results"""
        logger.info("=== Dataset Overlap Analysis Results ===")
        
        for overlap_type, data in results.items():
            if overlap_type in ['train_val_overlap', 'train_test_overlap', 'val_test_overlap'] and data:
                logger.info(f"\n{overlap_type.replace('_', '-').title()}:")
                logger.info(f"  Samples checked: {data.get('total_checked', 0)}")
                logger.info(f"  High overlap (>50%): {data.get('overlap_count', 0)} ({data.get('overlap_percentage', 0):.1%})")
                logger.info(f"  Exact matches: {data.get('exact_matches', 0)} ({data.get('exact_match_percentage', 0):.1%})")
        
        if results.get('recommendations'):
            logger.info("\nRecommendations:")
            for rec in results['recommendations']:
                logger.info(f"  • {rec}")
    
    def _print_sequence_validation_results(self, results: Dict[str, Any]):
        """Print formatted sequence validation results"""
        logger.info("=== Non-overlapping Sequence Validation ===")
        
        stats = results.get('statistics', {})
        logger.info(f"Total samples: {stats.get('total_samples', 'Unknown')}")
        logger.info(f"Expected stride: {stats.get('expected_stride', 'Unknown')}")
        logger.info(f"Actual stride: {stats.get('most_common_stride', 'Unknown')}")
        logger.info(f"Stride consistent: {stats.get('stride_consistency', 'Unknown')}")
        
        if 'expected_samples' in stats:
            logger.info(f"Expected samples: {stats['expected_samples']}")
            logger.info(f"Actual samples: {stats['actual_samples']}")
            logger.info(f"Sample efficiency: {stats.get('sample_efficiency', 0):.1%}")
        
        if results.get('issues_found'):
            logger.info("\nIssues found:")
            for issue in results['issues_found']:
                logger.info(f"  ❌ {issue}")
        else:
            logger.info("\n✅ All sequence validation checks passed")
        
        if results.get('recommendations'):
            logger.info("\nRecommendations:")
            for rec in results['recommendations']:
                logger.info(f"  • {rec}")


# Convenience functions for easy use
def validate_training_run(training_history: Dict[str, List[float]], 
                         train_data: List[str] = None,
                         val_data: List[str] = None,
                         test_data: List[str] = None,
                         dataset: Any = None,
                         seq_length: int = 512,
                         output_dir: str = None,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive validation of a training run including metrics, data leakage, and sequence validation.
    
    Args:
        training_history: Dictionary with training metrics over time
        train_data: Optional training dataset sequences for leakage detection
        val_data: Optional validation dataset sequences for leakage detection
        test_data: Optional test dataset sequences for leakage detection
        dataset: Optional dataset object for sequence validation
        seq_length: Sequence length for analysis
        output_dir: Optional directory to save validation report
        verbose: Whether to print detailed results
        
    Returns:
        Comprehensive validation results
    """
    if verbose:
        logger.info("Starting comprehensive training run validation...")
    
    validator = MetricsValidator()
    
    # 1. Validate training metrics
    metrics_results = validator.validate_training_metrics(training_history, verbose=verbose)
    
    # 2. Test perplexity calculation
    perplexity_results = validator.test_perplexity_calculation(verbose=verbose)
    
    # 3. Check for data leakage if data provided
    leakage_results = None
    if train_data and val_data:
        leakage_results = validator.check_dataset_overlap(
            train_data, val_data, test_data, seq_length=seq_length, verbose=verbose
        )
    
    # 4. Validate sequence non-overlap if dataset provided
    sequence_results = None
    if dataset:
        sequence_results = validator.validate_non_overlapping_sequences(
            dataset, seq_length, verbose=verbose
        )
    
    # 5. Evaluate baseline if possible
    baseline_results = validator.evaluate_gpt2_baseline(verbose=verbose)
    
    # Compile comprehensive results
    comprehensive_results = {
        'metrics_validation': metrics_results,
        'perplexity_validation': perplexity_results,
        'data_leakage_analysis': leakage_results,
        'sequence_validation': sequence_results,
        'baseline_evaluation': baseline_results,
        'overall_status': 'PASSED',
        'critical_issues': [],
        'recommendations': []
    }
    
    # Determine overall status
    critical_issues = []
    
    if not metrics_results.get('metrics_consistency', True):
        critical_issues.append("Training metrics show inconsistencies")
    
    if not perplexity_results.get('all_tests_passed', True):
        critical_issues.append("Perplexity calculation tests failed")
    
    if leakage_results:
        train_val = leakage_results.get('train_val_overlap', {})
        if train_val.get('exact_match_percentage', 0) > 0.01:
            critical_issues.append("Data leakage detected between train and validation sets")
    
    if sequence_results:
        if not sequence_results.get('is_non_overlapping', True):
            critical_issues.append("Sequences are overlapping (data leakage risk)")
    
    comprehensive_results['critical_issues'] = critical_issues
    comprehensive_results['overall_status'] = 'FAILED' if critical_issues else 'PASSED'
    
    # Compile all recommendations
    all_recommendations = []
    for result_key in ['metrics_validation', 'data_leakage_analysis', 'sequence_validation']:
        result = comprehensive_results.get(result_key)
        if result and 'recommendations' in result:
            all_recommendations.extend(result['recommendations'])
    
    comprehensive_results['recommendations'] = list(set(all_recommendations))  # Remove duplicates
    
    # Save report if output directory specified
    if output_dir:
        validator.generate_comprehensive_report(output_dir)
    
    if verbose:
        logger.info(f"\n=== COMPREHENSIVE VALIDATION SUMMARY ===")
        logger.info(f"Overall Status: {comprehensive_results['overall_status']}")
        
        if critical_issues:
            logger.info("Critical Issues:")
            for issue in critical_issues:
                logger.info(f"  ❌ {issue}")
        else:
            logger.info("✅ No critical issues found")
        
        if comprehensive_results['recommendations']:
            logger.info("\nKey Recommendations:")
            for rec in comprehensive_results['recommendations'][:5]:  # Show top 5
                logger.info(f"  • {rec}")
    
    return comprehensive_results


def run_comprehensive_validation_suite(training_history: Dict[str, List[float]] = None,
                                      train_data: List[str] = None,
                                      val_data: List[str] = None,
                                      test_data: List[str] = None,
                                      dataset: Any = None,
                                      dataset_loader: Any = None,
                                      seq_length: int = 512,
                                      device: str = 'cpu',
                                      output_dir: str = None,
                                      verbose: bool = True) -> Dict[str, Any]:
    """
    Run the complete validation suite with all available tests.
    
    This function runs all validation tests that are possible given the provided data:
    - Perplexity calculation verification
    - Training metrics validation (if training_history provided)
    - Data leakage detection (if train/val data provided)
    - Sequence overlap validation (if dataset provided)
    - Baseline model evaluation (if dataset_loader provided)
    
    Args:
        training_history: Optional training metrics history
        train_data: Optional training dataset sequences
        val_data: Optional validation dataset sequences
        test_data: Optional test dataset sequences
        dataset: Optional dataset object for sequence validation
        dataset_loader: Optional data loader for baseline evaluation
        seq_length: Sequence length for analysis
        device: Device for baseline evaluation
        output_dir: Optional directory to save comprehensive report
        verbose: Whether to print detailed results
        
    Returns:
        Complete validation suite results
    """
    if verbose:
        logger.info("🚀 Running Comprehensive SpectralLLM Validation Suite")
        logger.info("=" * 60)
    
    validator = MetricsValidator()
    suite_results = {
        'suite_version': '1.0.0',
        'tests_run': [],
        'tests_passed': 0,
        'tests_failed': 0,
        'critical_failures': [],
        'all_results': {}
    }
    
    # Test 1: Perplexity Calculation Verification (always run)
    if verbose:
        logger.info("\n📊 Test 1: Perplexity Calculation Verification")
    
    perplexity_results = validator.test_perplexity_calculation(verbose=verbose)
    suite_results['tests_run'].append('perplexity_verification')
    suite_results['all_results']['perplexity_verification'] = perplexity_results
    
    if perplexity_results.get('all_tests_passed', False):
        suite_results['tests_passed'] += 1
    else:
        suite_results['tests_failed'] += 1
        suite_results['critical_failures'].append('Perplexity calculation verification failed')
    
    # Test 2: Training Metrics Validation
    if training_history:
        if verbose:
            logger.info("\n📈 Test 2: Training Metrics Validation")
        
        metrics_results = validator.validate_training_metrics(training_history, verbose=verbose)
        suite_results['tests_run'].append('metrics_validation')
        suite_results['all_results']['metrics_validation'] = metrics_results
        
        if metrics_results.get('metrics_consistency', False):
            suite_results['tests_passed'] += 1
        else:
            suite_results['tests_failed'] += 1
            if metrics_results.get('issues_found'):
                suite_results['critical_failures'].extend(metrics_results['issues_found'])
    
    # Test 3: Data Leakage Detection
    if train_data and val_data:
        if verbose:
            logger.info("\n🔍 Test 3: Data Leakage Detection")
        
        leakage_results = validator.check_dataset_overlap(
            train_data, val_data, test_data, seq_length=seq_length, verbose=verbose
        )
        suite_results['tests_run'].append('data_leakage_detection')
        suite_results['all_results']['data_leakage_detection'] = leakage_results
        
        # Check for critical leakage
        train_val = leakage_results.get('train_val_overlap', {})
        if train_val.get('exact_match_percentage', 0) < 0.01:  # Less than 1% exact matches is good
            suite_results['tests_passed'] += 1
        else:
            suite_results['tests_failed'] += 1
            suite_results['critical_failures'].append(
                f"Data leakage detected: {train_val.get('exact_match_percentage', 0):.1%} exact matches"
            )
    
    # Test 4: Sequence Overlap Validation
    if dataset:
        if verbose:
            logger.info("\n🔗 Test 4: Non-overlapping Sequence Validation")
        
        sequence_results = validator.validate_non_overlapping_sequences(
            dataset, seq_length, verbose=verbose
        )
        suite_results['tests_run'].append('sequence_validation')
        suite_results['all_results']['sequence_validation'] = sequence_results
        
        if sequence_results.get('is_non_overlapping', False) and sequence_results.get('stride_correct', False):
            suite_results['tests_passed'] += 1
        else:
            suite_results['tests_failed'] += 1
            if sequence_results.get('issues_found'):
                suite_results['critical_failures'].extend(sequence_results['issues_found'])
    
    # Test 5: Baseline Model Evaluation
    if dataset_loader or not any([training_history, train_data, dataset]):  # Run if no other data or if loader provided
        if verbose:
            logger.info("\n🤖 Test 5: Baseline Model Evaluation")
        
        baseline_results = validator.evaluate_gpt2_baseline(
            dataset_loader=dataset_loader, seq_length=seq_length, device=device, verbose=verbose
        )
        suite_results['tests_run'].append('baseline_evaluation')
        suite_results['all_results']['baseline_evaluation'] = baseline_results
        
        if 'error' not in baseline_results:
            suite_results['tests_passed'] += 1
        else:
            suite_results['tests_failed'] += 1
    
    # Calculate overall results
    total_tests = len(suite_results['tests_run'])
    pass_rate = suite_results['tests_passed'] / total_tests if total_tests > 0 else 0
    suite_results['pass_rate'] = pass_rate
    suite_results['overall_status'] = 'PASSED' if pass_rate >= 0.8 and not suite_results['critical_failures'] else 'FAILED'
    
    # Generate comprehensive report
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, 'validation_suite_report.json')
            with open(report_path, 'w') as f:
                json.dump(suite_results, f, indent=2, default=str)
            if verbose:
                logger.info(f"📄 Comprehensive report saved to: {report_path}")
        except Exception as e:
            if verbose:
                logger.warning(f"Could not save report: {e}")
    
    # Print final summary
    if verbose:
        logger.info("\n" + "=" * 60)
        logger.info("🏁 VALIDATION SUITE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Tests Run: {total_tests}")
        logger.info(f"Tests Passed: {suite_results['tests_passed']}")
        logger.info(f"Tests Failed: {suite_results['tests_failed']}")
        logger.info(f"Pass Rate: {pass_rate:.1%}")
        logger.info(f"Overall Status: {suite_results['overall_status']}")
        
        if suite_results['critical_failures']:
            logger.info("\n❌ Critical Issues:")
            for failure in suite_results['critical_failures']:
                logger.info(f"  • {failure}")
        else:
            logger.info("\n✅ No critical issues detected!")
        
        logger.info("\n🎯 Tests Completed:")
        for test in suite_results['tests_run']:
            status = "✅" if test in [k for k, v in suite_results['all_results'].items() 
                                   if not v.get('error') and (v.get('all_tests_passed', True) or 
                                   v.get('metrics_consistency', True) or 
                                   v.get('is_non_overlapping', True))] else "❌"
            logger.info(f"  {status} {test.replace('_', ' ').title()}")
    
    return suite_results


__all__ = [
    'MetricsValidator',
    'validate_training_run',
    'run_comprehensive_validation_suite'
] 