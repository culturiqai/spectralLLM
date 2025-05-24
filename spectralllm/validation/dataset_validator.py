#!/usr/bin/env python3
"""
Dataset Validation Module
=========================

Validates dataset integrity, checks for data leakage, and verifies sequence construction.
Based on insights from verify_perplexity.py.
"""

import torch
import logging
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm


class DatasetValidator:
    """
    Validates dataset integrity and construction quality.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the dataset validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def check_dataset_overlap(self, train_dataset: Any, val_dataset: Any, 
                            max_samples: int = 500, min_overlap_tokens: int = 10) -> Dict[str, float]:
        """
        Check for potential data leakage between training and validation sets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            max_samples: Maximum number of samples to check
            min_overlap_tokens: Minimum tokens for overlap detection
            
        Returns:
            Dictionary with overlap statistics
        """
        self.logger.info("Checking for dataset overlap between train and validation...")
        
        # Sample subsets to check
        val_samples = min(max_samples, len(val_dataset))
        train_samples = min(max_samples * 10, len(train_dataset))  # Check more training samples
        
        self.logger.info(f"Checking {val_samples} validation samples against {train_samples} training samples...")
        
        overlap_count = 0
        high_overlap_samples = []
        
        # For each validation sample, check for matches in training set
        for i in tqdm(range(val_samples), desc="Checking overlap"):
            try:
                # Handle tuple format
                val_item = val_dataset[i]
                if isinstance(val_item, tuple):
                    val_x = val_item[0]
                else:
                    val_x = val_item
                
                if isinstance(val_x, torch.Tensor):
                    val_ids = val_x.tolist()
                else:
                    val_ids = val_x
            except Exception:
                continue
            
            max_overlap = 0
            best_match_idx = -1
            
            for j in range(train_samples):
                try:
                    # Handle tuple format
                    train_item = train_dataset[j]
                    if isinstance(train_item, tuple):
                        train_x = train_item[0]
                    else:
                        train_x = train_item
                    
                    if isinstance(train_x, torch.Tensor):
                        train_ids = train_x.tolist()
                    else:
                        train_ids = train_x
                    
                    # Check for exact subsequence matches
                    overlap_score = self._calculate_overlap_score(val_ids, train_ids, min_overlap_tokens)
                    
                    if overlap_score > max_overlap:
                        max_overlap = overlap_score
                        best_match_idx = j
                        
                except Exception as e:
                    continue
            
            if max_overlap > 0.5:  # Consider high overlap if more than 50% tokens match
                overlap_count += 1
                high_overlap_samples.append({
                    "val_sample_id": i,
                    "train_sample_id": best_match_idx,
                    "overlap_score": max_overlap
                })
        
        overlap_percentage = overlap_count / val_samples
        
        self.logger.info(f"Found {overlap_count} validation samples with >50% overlap ({overlap_percentage:.2%})")
        
        if high_overlap_samples:
            self.logger.info("Top 5 highest overlap samples:")
            for sample in sorted(high_overlap_samples, key=lambda x: x["overlap_score"], reverse=True)[:5]:
                self.logger.info(f"Val {sample['val_sample_id']} -> Train {sample['train_sample_id']}: {sample['overlap_score']:.2%} overlap")
        
        # Assessment
        if overlap_percentage > 0.05:
            self.logger.warning("⚠️ WARNING: Significant dataset overlap detected (>5%)!")
        elif overlap_percentage > 0.01:
            self.logger.warning("⚠️ CAUTION: Some dataset overlap detected (>1%)")
        else:
            self.logger.info("✅ Dataset overlap within acceptable limits (<1%)")
        
        return {
            'overlap_percentage': overlap_percentage,
            'overlap_count': overlap_count,
            'total_checked': val_samples,
            'high_overlap_samples': high_overlap_samples[:5],  # Top 5
            'assessment': 'high' if overlap_percentage > 0.05 else 'medium' if overlap_percentage > 0.01 else 'low'
        }
    
    def _calculate_overlap_score(self, seq1: List[int], seq2: List[int], 
                               min_tokens: int = 10) -> float:
        """
        Calculate overlap score between two sequences.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            min_tokens: Minimum tokens for subsequence matching
            
        Returns:
            Overlap score between 0 and 1
        """
        if len(seq1) < min_tokens or len(seq2) < min_tokens:
            return 0.0
        
        # Convert to strings for efficient substring matching
        str1 = str(seq1)
        str2 = str(seq2)
        
        # Find longest common subsequence
        max_match_length = 0
        
        # Check for exact subsequence matches of at least min_tokens
        for i in range(len(seq1) - min_tokens + 1):
            subseq = str(seq1[i:i+min_tokens])
            if subseq in str2:
                # Found a match, try to extend it
                extended_length = min_tokens
                while (i + extended_length < len(seq1) and 
                       str(seq1[i:i+extended_length+1]) in str2):
                    extended_length += 1
                max_match_length = max(max_match_length, extended_length)
        
        # Calculate overlap as percentage of shorter sequence
        shorter_length = min(len(seq1), len(seq2))
        return max_match_length / shorter_length if shorter_length > 0 else 0.0
    
    def verify_sequence_construction(self, dataset: Any, expected_stride: int = None,
                                   expected_seq_length: int = None) -> Dict[str, Any]:
        """
        Verify that sequences are constructed correctly (non-overlapping, proper stride).
        
        Args:
            dataset: Dataset to verify
            expected_stride: Expected stride between sequences
            expected_seq_length: Expected sequence length
            
        Returns:
            Dictionary with verification results
        """
        self.logger.info("Verifying sequence construction...")
        
        results = {
            'total_samples': len(dataset),
            'stride_verification': {},
            'length_verification': {},
            'construction_quality': 'unknown'
        }
        
        # Check sequence lengths
        if expected_seq_length:
            length_mismatches = 0
            sample_size = min(100, len(dataset))
            
            for i in range(sample_size):
                try:
                    x, y = dataset[i]
                    if hasattr(x, '__len__'):
                        actual_length = len(x)
                        if actual_length != expected_seq_length:
                            length_mismatches += 1
                except Exception:
                    continue
            
            length_error_rate = length_mismatches / sample_size
            results['length_verification'] = {
                'expected_length': expected_seq_length,
                'error_rate': length_error_rate,
                'samples_checked': sample_size,
                'passed': length_error_rate < 0.01
            }
        
        # Check stride if dataset has indexable access to underlying data
        if hasattr(dataset, 'get_dataset_stats'):
            try:
                stats = dataset.get_dataset_stats()
                if 'stride' in stats and 'seq_length' in stats:
                    actual_stride = stats['stride']
                    actual_seq_length = stats['seq_length']
                    overlap_ratio = 1.0 - (actual_stride / actual_seq_length) if actual_seq_length > 0 else 0.0
                    
                    results['stride_verification'] = {
                        'actual_stride': actual_stride,
                        'actual_seq_length': actual_seq_length,
                        'overlap_ratio': overlap_ratio,
                        'is_non_overlapping': overlap_ratio < 0.01,
                        'passed': True
                    }
                    
                    if expected_stride and actual_stride != expected_stride:
                        results['stride_verification']['expected_stride'] = expected_stride
                        results['stride_verification']['stride_matches'] = False
                        results['stride_verification']['passed'] = False
            except Exception as e:
                self.logger.warning(f"Could not verify stride: {e}")
        
        # Overall assessment
        length_passed = results['length_verification'].get('passed', True)
        stride_passed = results['stride_verification'].get('passed', True)
        
        if length_passed and stride_passed:
            results['construction_quality'] = 'excellent'
            self.logger.info("✅ Sequence construction verification PASSED")
        elif length_passed or stride_passed:
            results['construction_quality'] = 'good'
            self.logger.warning("⚠️ Sequence construction has minor issues")
        else:
            results['construction_quality'] = 'poor'
            self.logger.error("❌ Sequence construction verification FAILED")
        
        return results
    
    def validate_training_steps(self, dataset_or_sizes: Any, batch_size: int = None, 
                              expected_steps: int = None) -> Dict[str, Any]:
        """
        Validate expected number of training steps per epoch or analyze step reduction over time.
        
        Args:
            dataset_or_sizes: Training dataset or list of epoch sizes for reduction analysis
            batch_size: Batch size for training (required if dataset provided)
            expected_steps: Expected steps per epoch (optional)
            
        Returns:
            Dictionary with step validation results
        """
        # Handle case where dataset_or_sizes is a list of epoch sizes (for reduction analysis)
        if isinstance(dataset_or_sizes, list):
            self.logger.info("Analyzing training step reduction over time...")
            epoch_sizes = dataset_or_sizes
            
            if len(epoch_sizes) < 2:
                return {
                    'is_reducing': False,
                    'reduction_rate': 0.0,
                    'total_reduction': 0.0,
                    'insufficient_data': True
                }
            
            initial_size = epoch_sizes[0]
            final_size = epoch_sizes[-1]
            total_reduction = initial_size - final_size
            reduction_rate = total_reduction / initial_size if initial_size > 0 else 0.0
            
            # Check if generally decreasing
            decreasing_count = 0
            for i in range(1, len(epoch_sizes)):
                if epoch_sizes[i] < epoch_sizes[i-1]:
                    decreasing_count += 1
            
            is_reducing = decreasing_count >= len(epoch_sizes) // 2
            
            return {
                'is_reducing': is_reducing,
                'reduction_rate': reduction_rate,
                'total_reduction': total_reduction,
                'epoch_sizes': epoch_sizes,
                'initial_size': initial_size,
                'final_size': final_size,
                'decreasing_epochs': decreasing_count
            }
        
        # Original dataset validation logic
        self.logger.info("Validating training steps with current dataset...")
        
        if batch_size is None:
            batch_size = 32  # Default batch size
        
        total_samples = len(dataset_or_sizes)
        calculated_steps = (total_samples + batch_size - 1) // batch_size  # Ceiling division
        
        results = {
            'total_samples': total_samples,
            'batch_size': batch_size,
            'calculated_steps': calculated_steps,
            'expected_steps': expected_steps,
            'validation_passed': True
        }
        
        if expected_steps:
            step_difference = abs(calculated_steps - expected_steps)
            step_ratio = calculated_steps / expected_steps if expected_steps > 0 else 0
            
            results.update({
                'step_difference': step_difference,
                'step_ratio': step_ratio,
                'matches_expected': step_difference <= 1  # Allow 1 step difference due to rounding
            })
            
            if 0.45 <= step_ratio <= 0.55:
                self.logger.info("✅ Step count is approximately half after non-overlapping fix")
                results['improvement_detected'] = True
            elif step_ratio < 0.45:
                self.logger.info("✅ Step count significantly reduced (better than expected)")
                results['improvement_detected'] = True
            else:
                self.logger.warning("⚠️ Step count not reduced as expected")
                results['improvement_detected'] = False
        
        self.logger.info(f"Dataset configuration: batch_size={batch_size}")
        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info(f"Calculated steps per epoch: {calculated_steps}")
        
        return results
    
    def save_validation_results(self, results: Dict[str, Any], output_dir: str, 
                              timestamp: str = None) -> str:
        """
        Save validation results to file.
        
        Args:
            results: Validation results dictionary
            output_dir: Output directory
            timestamp: Optional timestamp (will generate if not provided)
            
        Returns:
            Path to saved results file
        """
        if not timestamp:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"dataset_validation_{timestamp}.json")
        
        # Add metadata
        results_with_metadata = {
            'validation_timestamp': timestamp,
            'validation_results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        self.logger.info(f"Validation results saved to {results_file}")
        return results_file
    
    def compare_datasets(self, old_dataset: Any, new_dataset: Any, 
                        description: str = "dataset comparison") -> Dict[str, Any]:
        """
        Compare two datasets to validate improvements.
        
        Args:
            old_dataset: Original dataset
            new_dataset: Updated dataset
            description: Description of the comparison
            
        Returns:
            Comparison results
        """
        self.logger.info(f"Comparing datasets: {description}")
        
        comparison = {
            'description': description,
            'old_dataset_size': len(old_dataset),
            'new_dataset_size': len(new_dataset),
            'size_ratio': len(new_dataset) / len(old_dataset) if len(old_dataset) > 0 else 0,
        }
        
        # Check if new dataset has fewer samples (expected with non-overlapping)
        if comparison['size_ratio'] < 0.6:
            comparison['assessment'] = 'significant_reduction'
            self.logger.info("✅ Significant reduction in dataset size (likely due to non-overlapping fix)")
        elif comparison['size_ratio'] < 0.8:
            comparison['assessment'] = 'moderate_reduction'
            self.logger.info("✅ Moderate reduction in dataset size")
        else:
            comparison['assessment'] = 'minimal_change'
            self.logger.info("⚠️ Minimal change in dataset size")
        
        return comparison
    
    def detect_subsequence_overlap(self, train_texts: List[str], val_texts: List[str], 
                                 min_length: int = 3) -> Dict[str, Any]:
        """
        Detect subsequence overlaps between training and validation texts.
        
        Args:
            train_texts: List of training text strings
            val_texts: List of validation text strings
            min_length: Minimum subsequence length to consider
            
        Returns:
            Dictionary with overlap analysis results
        """
        self.logger.info("Detecting subsequence overlaps...")
        
        overlapping_subsequences = []
        total_overlap_score = 0.0
        
        for val_idx, val_text in enumerate(val_texts):
            val_words = val_text.lower().split()
            
            for train_idx, train_text in enumerate(train_texts):
                train_words = train_text.lower().split()
                
                # Find common subsequences
                for i in range(len(val_words) - min_length + 1):
                    val_subseq = val_words[i:i + min_length]
                    val_subseq_str = ' '.join(val_subseq)
                    
                    for j in range(len(train_words) - min_length + 1):
                        train_subseq = train_words[j:j + min_length]
                        train_subseq_str = ' '.join(train_subseq)
                        
                        if val_subseq_str == train_subseq_str:
                            overlap_info = {
                                'val_idx': val_idx,
                                'train_idx': train_idx,
                                'subsequence': val_subseq_str,
                                'val_position': i,
                                'train_position': j,
                                'length': len(val_subseq)
                            }
                            overlapping_subsequences.append(overlap_info)
                            total_overlap_score += len(val_subseq)
        
        # Calculate overall overlap score
        total_possible_overlaps = sum(len(text.split()) for text in val_texts)
        overlap_score = total_overlap_score / total_possible_overlaps if total_possible_overlaps > 0 else 0.0
        
        result = {
            'overlapping_subsequences': overlapping_subsequences,
            'overlap_score': overlap_score,
            'total_overlaps': len(overlapping_subsequences),
            'assessment': 'high' if overlap_score > 0.1 else 'medium' if overlap_score > 0.05 else 'low'
        }
        
        self.logger.info(f"Found {len(overlapping_subsequences)} subsequence overlaps with score {overlap_score:.4f}")
        return result
    
    def calculate_data_quality_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate data quality metrics for a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        if not texts:
            return {'avg_length': 0, 'length_std': 0, 'vocab_diversity': 0, 'repetition_ratio': 1.0}
        
        # Length statistics
        lengths = [len(text) for text in texts]
        avg_length = sum(lengths) / len(lengths)
        length_std = (sum((l - avg_length) ** 2 for l in lengths) / len(lengths)) ** 0.5
        
        # Vocabulary diversity
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        unique_words = set(all_words)
        vocab_diversity = len(unique_words) / len(all_words) if all_words else 0
        
        # Repetition ratio (how much text is repeated)
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        total_words = len(all_words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        repetition_ratio = repeated_words / total_words if total_words > 0 else 0
        
        return {
            'avg_length': avg_length,
            'length_std': length_std,
            'vocab_diversity': vocab_diversity,
            'repetition_ratio': repetition_ratio
        }
    
    def generate_validation_report(self, train_texts: List[str], val_texts: List[str], 
                                 tokenizer: Any = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            train_texts: Training text data
            val_texts: Validation text data
            tokenizer: Optional tokenizer for advanced analysis
            
        Returns:
            Comprehensive validation report
        """
        self.logger.info("Generating comprehensive validation report...")
        
        # Data leakage analysis
        leakage_result = self.detect_data_leakage(train_texts, val_texts)
        
        # Data quality analysis
        train_quality = self.calculate_data_quality_metrics(train_texts)
        val_quality = self.calculate_data_quality_metrics(val_texts)
        
        # Subsequence overlap analysis
        overlap_result = self.detect_subsequence_overlap(train_texts, val_texts)
        
        # Generate summary
        total_issues = 0
        issues = []
        
        if leakage_result['assessment'] != 'low':
            total_issues += 1
            issues.append(f"Data leakage detected: {leakage_result['assessment']}")
        
        if overlap_result['assessment'] != 'low':
            total_issues += 1
            issues.append(f"Subsequence overlap: {overlap_result['assessment']}")
        
        if train_quality['repetition_ratio'] > 0.3:
            total_issues += 1
            issues.append("High repetition ratio in training data")
        
        # Calculate severity score
        severity_score = min(total_issues / 3.0, 1.0)  # Normalize to 0-1
        pass_rate = 1.0 - severity_score
        
        # Generate recommendations
        recommendations = []
        if total_issues == 0:
            recommendations.append("Dataset validation passed - no major issues detected")
        else:
            if leakage_result['assessment'] != 'low':
                recommendations.append("Consider removing overlapping samples between train/val sets")
            if overlap_result['assessment'] != 'low':
                recommendations.append("Review subsequence overlaps and consider data preprocessing")
            if train_quality['repetition_ratio'] > 0.3:
                recommendations.append("Consider deduplication to reduce repetitive content")
        
        report = {
            'summary': f"Validation completed with {total_issues} issues detected",
            'data_leakage': leakage_result,
            'data_quality': {
                'train_quality': train_quality,
                'val_quality': val_quality
            },
            'subsequence_overlap': overlap_result,
            'total_issues': total_issues,
            'severity_score': severity_score,
            'pass_rate': pass_rate,
            'recommendations': recommendations,
            'status': 'passed' if total_issues == 0 else 'warning' if total_issues < 3 else 'failed'
        }
        
        self.logger.info(f"Validation report generated: {report['status']} with {total_issues} issues")
        return report

    # Method aliases for test compatibility
    def detect_data_leakage(self, train_dataset, val_dataset):
        """Alias for test compatibility"""
        return self.check_dataset_overlap(train_dataset, val_dataset)
    
    def verify_datasets(self, dataset):
        """Alias for test compatibility"""
        return self.verify_sequence_construction(dataset) 