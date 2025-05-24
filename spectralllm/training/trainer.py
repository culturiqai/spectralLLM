"""
SpectralLLM Training Infrastructure
==================================

Complete training, evaluation, and benchmarking system for SpectralLLM models.
"""

import time
import logging
import json
import os
import traceback
import math
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from tqdm import tqdm

# Import SpectralLLM components
from ..core.model import SignalLLM
from ..core.evolution import HRFEvoController
from ..core.config import Config

# Import performance profiling utilities
try:
    from ..utils.performance import PerformanceProfiler, get_profiler, profile_function, BenchmarkSuite
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False
    PerformanceProfiler = None

# Import MPS optimizations
try:
    from ..utils.mps_optimizations import optimize_for_mps
    MPS_AVAILABLE = True
except ImportError:
    MPS_AVAILABLE = False


def log_exception(e, context=""):
    """Enhanced error logging with traceback"""
    error_msg = f"ERROR in {context}: {str(e)}\n{traceback.format_exc()}"
    logging.error(error_msg)
    
    # Save to error log file
    error_log_path = "spectralllm_training_error.log"
    with open(error_log_path, "a") as f:
        f.write(f"\n{'='*50}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n{error_msg}\n")
    
    return error_msg


def create_optimized_scheduler(optimizer, total_steps: int, warmup_steps: int = 500, 
                             min_lr_ratio: float = 0.1):
    """
    Create optimized learning rate scheduler with cosine annealing and proper warmup.
    Based on train_optimized_lr.py improvements.
    
    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps (default: 500 for faster convergence)
        min_lr_ratio: Minimum LR as ratio of peak (default: 0.1 = 10% of peak LR)
    
    Returns:
        LambdaLR scheduler with optimized decay strategy
    """
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine annealing after warmup (better than linear decay)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


class EnhancedTextDataset(Dataset):
    """
    Enhanced dataset for text data with better validation and non-overlapping sequences.
    Implements improvements from WikiText-103 training script.
    """
    def __init__(self, texts: List[str], tokenizer: Any, 
                 seq_length: int = 512, stride: int = None,
                 validate_sequences: bool = True):
        """
        Initialize enhanced text dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer for encoding text
            seq_length: Maximum sequence length
            stride: Stride for sequences (None = non-overlapping)
            validate_sequences: Whether to validate dataset construction
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.validate_sequences = validate_sequences
        
        # Use non-overlapping sequences by default for better training
        if stride is None:
            stride = seq_length
        self.stride = stride
        
        logging.info(f"Creating enhanced dataset with seq_length={seq_length}, stride={stride}")
        
        # Encode all texts and create token list
        self.token_ids = []
        for text in tqdm(texts, desc="Tokenizing texts"):
            if hasattr(tokenizer, 'encode'):
                tokens = tokenizer.encode(text)
            else:
                # Fallback for custom tokenizers
                tokens = tokenizer(text)
                if hasattr(tokens, 'input_ids'):
                    tokens = tokens.input_ids
            
            if len(tokens) > 0:
                self.token_ids.extend(tokens)
        
        logging.info(f"Total tokens: {len(self.token_ids)}")
        
        # Create samples with specified stride
        self.samples = []
        for i in range(0, len(self.token_ids) - seq_length - 1, stride):
            self.samples.append(i)
        
        # Validate dataset construction if requested
        if validate_sequences:
            self._validate_dataset_construction()
        
        logging.info(f"Created {len(self.samples)} samples")
    
    def _validate_dataset_construction(self):
        """Validate dataset construction similar to WikiText-103 script"""
        # First validation: Check sample count
        expected_samples = (len(self.token_ids) - self.seq_length - 1) // self.stride
        actual_samples = len(self.samples)
        
        # Allow at most 1% difference due to edge effects
        margin = max(1, expected_samples * 0.01)
        assert abs(expected_samples - actual_samples) <= margin, \
            f"Sample count validation failed: expected ~{expected_samples}, got {actual_samples}"
        
        # Second validation: Check stride between samples
        if len(self.samples) >= 2:
            actual_stride = self.samples[1] - self.samples[0]
            assert actual_stride == self.stride, \
                f"Stride validation failed: expected {self.stride}, got {actual_stride}"
        
        # Third validation: Check sequence lengths
        if len(self.samples) > 0:
            sample_input, sample_target = self[0]
            assert len(sample_input) == self.seq_length, \
                f"Input sequence length validation failed: expected {self.seq_length}, got {len(sample_input)}"
            assert len(sample_target) == self.seq_length, \
                f"Target sequence length validation failed: expected {self.seq_length}, got {len(sample_target)}"
        
        logging.info(f"✓ Dataset validation passed: non-overlapping sequences (stride={self.stride})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.samples[idx]
        end_idx = start_idx + self.seq_length + 1
        tokens = self.token_ids[start_idx:end_idx]
        
        # Create input and target tensors
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'total_tokens': len(self.token_ids),
            'total_samples': len(self.samples),
            'seq_length': self.seq_length,
            'stride': self.stride,
            'overlap_ratio': 1.0 - (self.stride / self.seq_length) if self.seq_length > 0 else 0.0
        }


# Keep original TextDataset for backward compatibility
class TextDataset(EnhancedTextDataset):
    """
    Backward compatible TextDataset that uses enhanced implementation.
    """
    def __init__(self, texts: List[str], tokenizer: Any, 
                 seq_length: int = 512, stride: int = 256):
        # Use the enhanced dataset with overlapping sequences for compatibility
        super().__init__(texts, tokenizer, seq_length, stride, validate_sequences=False)


class SpectralTrainer:
    """
    Comprehensive trainer for SpectralLLM models with advanced features.
    Enhanced with better error handling and training monitoring.
    """
    
    def __init__(self, model: SignalLLM, config: Config, use_optimized_lr: bool = True,
                 enable_profiling: bool = True, enable_mps_optimization: bool = True):
        """
        Initialize the SpectralLLM trainer.
        
        Args:
            model: SpectralLLM model to train
            config: Training configuration
            use_optimized_lr: Whether to use optimized LR scheduling (default: True)
            enable_profiling: Whether to enable performance profiling (default: True)
            enable_mps_optimization: Whether to apply MPS optimizations if available (default: True)
        """
        self.model = model
        self.config = config
        self.use_optimized_lr = use_optimized_lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize performance profiler
        self.profiler = None
        if enable_profiling and PERFORMANCE_AVAILABLE:
            self.profiler = get_profiler()
            print("✅ Performance profiling enabled")
        elif enable_profiling:
            print("⚠️  Performance profiling requested but not available")
        
        # Apply MPS optimizations if available and requested
        if enable_mps_optimization and MPS_AVAILABLE and self.device.type in ['mps', 'cpu']:
            try:
                print("🚀 Applying MPS optimizations...")
                self.model = optimize_for_mps(self.model, self.device)
                print("✅ MPS optimizations applied successfully")
            except Exception as e:
                print(f"⚠️  MPS optimization failed: {e}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer with better parameters for language models (from train_optimized_lr.py)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=getattr(config, 'weight_decay', 0.01),
            betas=(0.9, 0.95),  # Better betas for language models
            eps=1e-8
        )
        
        # Setup scheduler - will be properly configured in train() method
        if use_optimized_lr:
            # Placeholder - will be set up with proper total_steps in train()
            self.scheduler = None
            self._scheduler_type = 'optimized'
        else:
            # Fallback to original scheduler
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.8, 
                patience=2
            )
            self._scheduler_type = 'plateau'
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # HRFEvo controller for basis evolution
        self.hrfevo_controller = None
        if hasattr(config, 'use_hrfevo') and config.use_hrfevo:
            self.hrfevo_controller = HRFEvoController(config)
        
        # Training statistics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.error_count = 0
        self.batch_count = 0
        
        # Enhanced memory tracking with performance profiler
        self.memory_stats = []
        if self.profiler:
            self.profiler.start_continuous_monitoring(interval=5.0)  # Monitor every 5 seconds
        
        print(f"✅ SpectralTrainer initialized on {self.device}")
        print(f"   Model parameters: {self.model.count_parameters():,}")
        print(f"   HRFEvo enabled: {self.hrfevo_controller is not None}")
        print(f"   Optimized LR scheduling: {self.use_optimized_lr}")
        print(f"   Performance profiling: {self.profiler is not None}")
        if self.use_optimized_lr:
            print(f"   Optimizer betas: (0.9, 0.95) for language model optimization")
    
    def _check_flag_files(self, output_dir: str) -> bool:
        """
        Check for flag files requesting model save and/or training stop.
        Returns True if training should stop.
        """
        flag_dir = os.path.join(output_dir, "..", "saved_models")
        os.makedirs(flag_dir, exist_ok=True)
        
        for filename in os.listdir(flag_dir):
            if filename.startswith("save_and_stop_") and filename.endswith(".flag"):
                flag_path = os.path.join(flag_dir, filename)
                
                try:
                    with open(flag_path, "r") as f:
                        flag_data = json.load(f)
                        
                    # Check if this flag is for our process
                    pid_to_stop = flag_data.get("pid_to_stop")
                    if pid_to_stop is not None and str(os.getpid()) != str(pid_to_stop):
                        continue
                        
                    # Save the model
                    timestamp = flag_data.get("timestamp", time.strftime("%Y%m%d_%H%M%S"))
                    save_path = os.path.join(flag_dir, f"model_save_{timestamp}.pt")
                    
                    logging.info(f"Flag file detected: {filename}. Saving model to {save_path}")
                    
                    # Save model checkpoint
                    self.save_checkpoint(save_path, epoch=0, metrics={'flag_save': True})
                    
                    # Create completion indicator
                    completion_path = f"{flag_path}.completed"
                    with open(completion_path, "w") as f:
                        json.dump({
                            "model_path": save_path,
                            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                            "stats": {
                                "batch_count": self.batch_count,
                                "error_count": self.error_count,
                                "train_losses": self.train_losses[-5:] if self.train_losses else []
                            }
                        }, f)
                        
                    # Check if we should stop training
                    if flag_data.get("command") == "save_and_stop":
                        logging.info("Stopping training as requested by flag file")
                        return True
                        
                    # Remove the flag file to prevent reprocessing
                    try:
                        os.remove(flag_path)
                    except:
                        logging.warning(f"Could not remove flag file {flag_path}")
                        
                except Exception as e:
                    log_exception(e, f"processing flag file {filename}")
        
        return False
    
    def _track_memory_usage(self):
        """Track memory usage for debugging"""
        if self.device.type == 'cuda' and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            self.memory_stats.append({
                'batch': self.batch_count,
                'allocated_mb': memory_allocated,
                'reserved_mb': memory_reserved
            })
            
            # Log every 100 batches in debug mode
            if os.environ.get('SPECTRALLLM_DEBUG', '0') == '1' and self.batch_count % 100 == 0:
                logging.debug(f"CUDA Memory: Allocated={memory_allocated:.2f}MB, Reserved={memory_reserved:.2f}MB")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int, 
                   output_dir: str = None) -> Dict[str, float]:
        """
        Train for one epoch with enhanced error handling and performance profiling.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            output_dir: Output directory for flag files
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        num_batches = 0
        
        # Setup progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            try:
                # Profile the entire batch processing if profiler available
                batch_context = None
                if self.profiler:
                    batch_context = self.profiler.profile_operation(
                        f"train_batch_epoch_{epoch}",
                        batch_idx=batch_idx,
                        epoch=epoch,
                        batch_size=input_ids.size(0),
                        seq_length=input_ids.size(1)
                    )
                    batch_context.__enter__()
                
                # Check for flag files every 10 batches
                if output_dir and batch_idx % 10 == 0:
                    if self._check_flag_files(output_dir):
                        logging.info("Training stopped by flag file")
                        break
                
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Track memory usage (enhanced with profiler integration)
                self._track_memory_usage()
                
                # Forward pass with profiling
                if self.profiler:
                    with self.profiler.profile_operation("forward_pass", batch_idx=batch_idx):
                        self.optimizer.zero_grad()
                        outputs = self.model(input_ids)
                else:
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids)
                
                # Calculate loss with profiling
                if self.profiler:
                    with self.profiler.profile_operation("loss_calculation", batch_idx=batch_idx):
                        loss = self.criterion(
                            outputs.view(-1, outputs.size(-1)), 
                            target_ids.view(-1)
                        )
                else:
                    loss = self.criterion(
                        outputs.view(-1, outputs.size(-1)), 
                        target_ids.view(-1)
                    )
                
                # Backward pass
                if self.profiler:
                    with self.profiler.profile_operation("backward_pass", batch_idx=batch_idx):
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        self.optimizer.step()
                else:
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.optimizer.step()
                
                # Step optimized scheduler per batch (not per epoch)
                if self._scheduler_type == 'optimized' and self.scheduler is not None:
                    self.scheduler.step()
                
                # Calculate metrics
                batch_size, seq_length = target_ids.shape
                tokens_in_batch = batch_size * seq_length
                
                # Weight loss by number of tokens for proper averaging
                total_loss += loss.item() * tokens_in_batch
                total_tokens += tokens_in_batch
                num_batches += 1
                self.batch_count += 1
                
                # Calculate accuracy
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=-1)
                    correct = (preds == target_ids).sum().item()
                    total_correct += correct
                    
                    batch_accuracy = correct / tokens_in_batch
                
                # Update progress bar with LR info
                avg_loss = total_loss / total_tokens
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                current_lr = (self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') 
                            else self.optimizer.param_groups[0]['lr'])
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.2f}',
                    'acc': f'{batch_accuracy:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                # Evolution step (if enabled)
                if self.hrfevo_controller and batch_idx % 50 == 0:
                    if self.profiler:
                        with self.profiler.profile_operation("evolution_step", batch_idx=batch_idx):
                            self._evolution_step(input_ids, target_ids)
                    else:
                        self._evolution_step(input_ids, target_ids)
                
                # Add custom metrics to profiler
                if self.profiler:
                    self.profiler.add_custom_metric("batch_loss", loss.item(), batch_idx=batch_idx, epoch=epoch)
                    self.profiler.add_custom_metric("batch_accuracy", batch_accuracy, batch_idx=batch_idx, epoch=epoch)
                    self.profiler.add_custom_metric("learning_rate", current_lr, batch_idx=batch_idx, epoch=epoch)
            
            except Exception as e:
                self.error_count += 1
                error_msg = log_exception(e, f"training batch {batch_idx}")
                logging.error(f"Error processing batch {batch_idx}: {error_msg}")
                
                # Save problematic tensors for debugging if in debug mode
                if os.environ.get('SPECTRALLLM_DEBUG', '0') == '1':
                    debug_dir = os.path.join(output_dir or ".", "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    try:
                        torch.save({
                            'inputs': input_ids.cpu() if 'input_ids' in locals() else None,
                            'targets': target_ids.cpu() if 'target_ids' in locals() else None,
                            'batch_idx': batch_idx,
                            'error': str(e)
                        }, os.path.join(debug_dir, f"error_batch_{batch_idx}.pt"))
                        logging.debug(f"Saved error debug info to {debug_dir}")
                    except:
                        logging.debug("Could not save debug tensors")
                
                # Continue with next batch
                continue
            
            finally:
                # Always clean up the profiling context
                if batch_context:
                    try:
                        batch_context.__exit__(None, None, None)
                    except:
                        pass
        
        # Calculate epoch metrics with proper weighting
        avg_epoch_loss = total_loss / max(total_tokens, 1)
        epoch_accuracy = total_correct / max(total_tokens, 1)
        
        self.train_losses.append(avg_epoch_loss)
        
        return {
            'train_loss': avg_epoch_loss,
            'train_perplexity': torch.exp(torch.tensor(avg_epoch_loss)).item(),
            'train_accuracy': epoch_accuracy,
            'processed_batches': num_batches,
            'error_count': self.error_count,
            'total_tokens': total_tokens
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on validation/test data with enhanced metrics.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(dataloader, desc="Evaluating"):
                try:
                    input_ids = input_ids.to(self.device)
                    target_ids = target_ids.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(input_ids)
                    
                    # Calculate loss
                    loss = self.criterion(
                        outputs.view(-1, outputs.size(-1)),
                        target_ids.view(-1)
                    )
                    
                    # Calculate metrics
                    batch_size, seq_length = target_ids.shape
                    tokens_in_batch = batch_size * seq_length
                    
                    # Weight by number of tokens
                    total_loss += loss.item() * tokens_in_batch
                    total_tokens += tokens_in_batch
                    num_batches += 1
                    
                    # Calculate accuracy
                    preds = torch.argmax(outputs, dim=-1)
                    correct = (preds == target_ids).sum().item()
                    total_correct += correct
                
                except Exception as e:
                    error_msg = log_exception(e, "evaluation batch")
                    logging.error(f"Error in evaluation: {error_msg}")
                    continue
        
        # Calculate properly weighted averages
        avg_loss = total_loss / max(total_tokens, 1)
        accuracy = total_correct / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity,
            'val_accuracy': accuracy,
            'processed_batches': num_batches,
            'total_tokens': total_tokens
        }
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
              num_epochs: int = 10, output_dir: str = None, 
              target_perplexity: float = None, warmup_steps: int = 500,
              min_lr_ratio: float = 0.1) -> Dict[str, List[float]]:
        """
        Complete training loop with enhanced monitoring and optimized LR scheduling.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            output_dir: Output directory for checkpoints and logs
            target_perplexity: Target perplexity for early stopping (optional)
            warmup_steps: Number of warmup steps for optimized scheduler
            min_lr_ratio: Minimum LR as ratio of peak for optimized scheduler
            
        Returns:
            Training history
        """
        print(f"🚀 Starting SpectralLLM training for {num_epochs} epochs")
        
        # Set up optimized scheduler if enabled
        if self.use_optimized_lr and self._scheduler_type == 'optimized':
            total_steps = num_epochs * len(train_loader)
            self.scheduler = create_optimized_scheduler(
                self.optimizer, 
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=min_lr_ratio
            )
            print(f"📊 Optimized LR scheduler configured:")
            print(f"   Total steps: {total_steps}")
            print(f"   Warmup steps: {warmup_steps}")
            print(f"   Min LR ratio: {min_lr_ratio} ({min_lr_ratio*100}% of peak)")
            if target_perplexity:
                print(f"🎯 Target perplexity: {target_perplexity}")
        
        history = {
            'train_loss': [],
            'train_perplexity': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_perplexity': [],
            'val_accuracy': []
        }
        
        # Track if target perplexity was reached
        target_reached = False
        target_epoch = None
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch, output_dir)
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_perplexity'].append(train_metrics['train_perplexity'])
            history['train_accuracy'].append(train_metrics['train_accuracy'])
            
            # Check if target perplexity reached on training
            if target_perplexity and train_metrics['train_perplexity'] < target_perplexity and not target_reached:
                target_reached = True
                target_epoch = epoch
                print(f"🎯 TARGET REACHED! Training perplexity {train_metrics['train_perplexity']:.2f} < {target_perplexity} at epoch {epoch}")
            
            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_perplexity'].append(val_metrics['val_perplexity'])
                history['val_accuracy'].append(val_metrics['val_accuracy'])
                
                # Check if target perplexity reached on validation
                if target_perplexity and val_metrics['val_perplexity'] < target_perplexity and not target_reached:
                    target_reached = True
                    target_epoch = epoch
                    print(f"🎯 TARGET REACHED! Validation perplexity {val_metrics['val_perplexity']:.2f} < {target_perplexity} at epoch {epoch}")
                
                # Learning rate scheduling
                if self._scheduler_type == 'plateau':
                    self.scheduler.step(val_metrics['val_loss'])
                # Optimized scheduler steps per batch, not per epoch
                
                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    if output_dir:
                        best_path = os.path.join(output_dir, f'best_model_epoch_{epoch}.pt')
                        self.save_checkpoint(best_path, epoch, val_metrics)
                
                # Enhanced logging with LR tracking
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: Train Loss={train_metrics['train_loss']:.4f}, "
                      f"Val Loss={val_metrics['val_loss']:.4f}, "
                      f"Val PPL={val_metrics['val_perplexity']:.2f}, "
                      f"LR={current_lr:.2e}, "
                      f"Errors={train_metrics['error_count']}")
            else:
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: Train Loss={train_metrics['train_loss']:.4f}, "
                      f"Train PPL={train_metrics['train_perplexity']:.2f}, "
                      f"LR={current_lr:.2e}, "
                      f"Errors={train_metrics['error_count']}")
        
        print("✅ Training completed!")
        if target_reached:
            print(f"🏆 Target perplexity {target_perplexity} reached at epoch {target_epoch}!")
        
        # Save final training statistics with target info
        if output_dir:
            stats_path = os.path.join(output_dir, "training_stats.json")
            final_stats = {
                'history': history,
                'total_errors': self.error_count,
                'total_batches': self.batch_count,
                'memory_stats': self.memory_stats[-10:] if self.memory_stats else [],  # Last 10 entries
                'target_perplexity': target_perplexity,
                'target_reached': target_reached,
                'target_epoch': target_epoch,
                'scheduler_type': self._scheduler_type,
                'optimizer_betas': (0.9, 0.95) if self.use_optimized_lr else (0.9, 0.999)
            }
            with open(stats_path, 'w') as f:
                json.dump(final_stats, f, indent=2)
            
            # Save comprehensive performance report if profiler is available
            if self.profiler:
                self.save_performance_report(output_dir)
        
        # Clean up profiler resources
        self.cleanup_profiler()
        
        return history
    
    def _evolution_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor):
        """
        Perform one step of basis function evolution with enhanced error handling.
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs
        """
        if self.hrfevo_controller is None:
            return
        
        def evaluate_basis(basis):
            """Evaluate a basis function by temporarily applying it to the model."""
            try:
                # Save current basis
                original_basis = self.model.current_basis.copy()
                
                # Apply new basis
                self.model.update_basis_function(basis)
                
                # Evaluate
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    loss = self.criterion(
                        outputs.view(-1, outputs.size(-1)),
                        target_ids.view(-1)
                    )
                    fitness = 1.0 / (loss.item() + 1e-8)
                
                # Restore original basis
                self.model.update_basis_function(original_basis)
                
                return {'fitness': fitness, 'loss': loss.item()}
            
            except Exception as e:
                log_exception(e, "basis evaluation")
                return {'fitness': 0.0, 'loss': float('inf')}
        
        # Run one evolution step
        try:
            stats = self.hrfevo_controller.evolve_generation(evaluate_basis)
            if hasattr(self, '_evo_step_count'):
                self._evo_step_count += 1
            else:
                self._evo_step_count = 1
                
            if self._evo_step_count % 10 == 0:
                print(f"  Evolution step {self._evo_step_count}: "
                      f"Best fitness = {stats['best_fitness']:.4f}")
        except Exception as e:
            log_exception(e, "evolution step")
            print(f"Evolution step failed: {e}")
    
    def benchmark_complexity(self, seq_lengths: List[int] = None, 
                           batch_size: int = 4, num_runs: int = 3) -> Dict:
        """
        Benchmark computational complexity at different sequence lengths.
        
        Args:
            seq_lengths: List of sequence lengths to test
            batch_size: Batch size for testing
            num_runs: Number of runs for averaging
            
        Returns:
            Benchmark results
        """
        if seq_lengths is None:
            seq_lengths = [64, 128, 256, 512, 1024]
        
        print(f"🔬 Benchmarking SpectralLLM complexity on {self.device}")
        
        results = {
            'seq_lengths': seq_lengths,
            'forward_times': [],
            'memory_usage': []
        }
        
        self.model.eval()
        
        with torch.no_grad():
            for seq_len in seq_lengths:
                print(f"  Testing sequence length: {seq_len}")
                
                # Create dummy input
                input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=self.device)
                
                # Warmup
                for _ in range(2):
                    _ = self.model(input_ids)
                
                # Benchmark timing
                times = []
                for _ in range(num_runs):
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start = time.time()
                    _ = self.model(input_ids)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    times.append(time.time() - start)
                
                avg_time = sum(times) / len(times)
                results['forward_times'].append(avg_time)
                
                # Memory usage (if CUDA)
                if self.device.type == 'cuda':
                    memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                    results['memory_usage'].append(memory_mb)
                    torch.cuda.reset_peak_memory_stats()
                else:
                    results['memory_usage'].append(0.0)
                
                print(f"    Time: {avg_time:.4f}s, Memory: {results['memory_usage'][-1]:.1f}MB")
        
        # Calculate complexity ratios
        if len(results['forward_times']) >= 2:
            ratios = []
            for i in range(1, len(seq_lengths)):
                theoretical_ratio = (seq_lengths[i] * seq_lengths[i]) / (seq_lengths[i-1] * seq_lengths[i-1])
                actual_ratio = results['forward_times'][i] / results['forward_times'][i-1]
                ratios.append(actual_ratio / theoretical_ratio)
            
            results['efficiency_ratios'] = ratios
            avg_efficiency = sum(ratios) / len(ratios)
            print(f"  Average efficiency vs O(n²): {avg_efficiency:.2f}x better")
        
        return results
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.hrfevo_controller:
            checkpoint['hrfevo_state'] = self.hrfevo_controller.save_state()
        
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if self.hrfevo_controller and 'hrfevo_state' in checkpoint:
            self.hrfevo_controller.load_state(checkpoint['hrfevo_state'])
        
        print(f"📁 Checkpoint loaded: {filepath}")
        return checkpoint['epoch'], checkpoint['metrics']
    
    def save_performance_report(self, output_dir: str, filename: str = "performance_report.json"):
        """
        Save comprehensive performance report from profiler.
        
        Args:
            output_dir: Directory to save the report
            filename: Name of the report file
        """
        if not self.profiler:
            print("⚠️  No profiler available - cannot save performance report")
            return
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, filename)
            
            # Generate comprehensive report
            self.profiler.save_report(report_path, include_raw_data=True)
            
            # Generate memory analysis
            memory_analysis = None
            try:
                from ..utils.performance import analyze_memory_patterns
                memory_analysis = analyze_memory_patterns(self.profiler)
            except ImportError:
                pass
            
            # Save additional analysis if available
            if memory_analysis:
                analysis_path = os.path.join(output_dir, "memory_analysis.json")
                with open(analysis_path, 'w') as f:
                    json.dump(memory_analysis, f, indent=2)
                print(f"📊 Memory analysis saved: {analysis_path}")
            
            print(f"📊 Performance report saved: {report_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("PERFORMANCE SUMMARY")
            print("="*60)
            
            summary = self.profiler.get_summary_report()
            
            if 'operations' in summary:
                print(f"Top operations by total time:")
                op_times = {}
                for op_name, stats in summary['operations'].items():
                    if 'total_time' in stats:
                        op_times[op_name] = stats['total_time']
                
                # Sort by total time and show top 5
                sorted_ops = sorted(op_times.items(), key=lambda x: x[1], reverse=True)[:5]
                for i, (op_name, total_time) in enumerate(sorted_ops, 1):
                    stats = summary['operations'][op_name]
                    print(f"  {i}. {op_name}:")
                    print(f"     Total: {total_time:.2f}s, Mean: {stats.get('mean_time', 0)*1000:.2f}ms, Count: {stats.get('count', 0)}")
            
            if 'peak_memory' in summary:
                print(f"\nPeak Memory Usage:")
                for memory_type, peak_mb in summary['peak_memory'].items():
                    print(f"  {memory_type}: {peak_mb:.2f} MB")
            
            if memory_analysis and memory_analysis['recommendations']:
                print(f"\nRecommendations:")
                for rec in memory_analysis['recommendations']:
                    print(f"  • {rec}")
            
            print("="*60)
            
        except Exception as e:
            print(f"⚠️  Failed to save performance report: {e}")
    
    def cleanup_profiler(self):
        """Clean up profiler resources"""
        if self.profiler:
            try:
                self.profiler.stop_continuous_monitoring()
            except:
                pass 