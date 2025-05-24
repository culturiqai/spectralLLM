#!/usr/bin/env python3
"""
Simple SpectralLLM Training Example
==================================

A minimal example showing how to train SpectralLLM on text data.
This is a simplified version suitable for learning and experimentation.
"""

import os
import time
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import spectralllm


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration"""
    
    def __init__(self, texts: List[str], seq_length: int = 128):
        self.seq_length = seq_length
        
        # Simple character-level tokenization
        all_text = ' '.join(texts)
        chars = sorted(list(set(all_text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # Convert to indices
        self.data = [self.char_to_idx[ch] for ch in all_text]
        
        # Create samples
        self.samples = []
        for i in range(0, len(self.data) - seq_length - 1, seq_length // 2):
            self.samples.append(i)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        start = self.samples[idx]
        sequence = self.data[start:start + self.seq_length + 1]
        return torch.tensor(sequence[:-1]), torch.tensor(sequence[1:])


def create_sample_data():
    """Create some sample text data for training"""
    texts = [
        "The future of artificial intelligence is bright and promising.",
        "Machine learning algorithms can process vast amounts of data efficiently.",
        "Neural networks learn complex patterns through iterative optimization.",
        "Deep learning has revolutionized computer vision and natural language processing.",
        "Transformers have become the foundation of modern language models.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Language models can generate coherent and contextually appropriate text.",
        "The development of AI requires careful consideration of ethical implications.",
        "Research in AI continues to push the boundaries of what's possible.",
        "Collaborative efforts between researchers advance the field of machine learning."
    ] * 10  # Repeat for more training data
    
    return texts


def train_spectralllm(args):
    """Train SpectralLLM on simple text data"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample data
    print("Creating sample dataset...")
    texts = create_sample_data()
    dataset = SimpleTextDataset(texts, args.seq_length)
    
    print(f"Dataset: {len(dataset)} samples, vocab size: {dataset.vocab_size}")
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Simple setup
    )
    
    # Create model
    print("Creating SpectralLLM model...")
    config = spectralllm.Config(
        vocab_size=dataset.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        harmonic_bases=args.harmonic_bases,
        max_seq_length=args.seq_length
    )
    
    model = spectralllm.SpectralLLM(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'ppl': f'{perplexity:.2f}'
            })
            
            # Optional: Generate sample text every 50 batches
            if batch_idx % 50 == 0 and batch_idx > 0:
                sample_text = generate_sample(model, dataset, device, max_length=50)
                print(f"\nSample generation: {sample_text}")
        
        # Epoch summary
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    # Save model
    if args.save_model:
        save_path = f"spectralllm_simple_{args.embed_dim}d_{args.num_layers}l.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'char_to_idx': dataset.char_to_idx,
            'idx_to_char': dataset.idx_to_char
        }, save_path)
        print(f"Model saved to {save_path}")
    
    # Final generation example
    print("\n" + "="*50)
    print("FINAL TEXT GENERATION EXAMPLES:")
    print("="*50)
    
    for prompt in ["The future", "Machine learning", "Neural networks"]:
        generated = generate_sample(model, dataset, device, prompt=prompt, max_length=100)
        print(f"Prompt: '{prompt}' -> {generated}")
    
    print("\n🎉 Training completed!")


def generate_sample(model, dataset, device, prompt=None, max_length=50):
    """Generate a sample text from the model"""
    model.eval()
    
    if prompt:
        # Use provided prompt
        sequence = [dataset.char_to_idx.get(ch, 0) for ch in prompt]
    else:
        # Start with a random character
        sequence = [torch.randint(0, dataset.vocab_size, (1,)).item()]
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            input_ids = torch.tensor([sequence], device=device)
            
            # Get predictions
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            sequence.append(next_token)
            
            # Stop if we reach max context length
            if len(sequence) >= model.config.max_seq_length:
                break
    
    # Convert back to text
    generated_text = ''.join([dataset.idx_to_char.get(idx, '?') for idx in sequence])
    model.train()
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Simple SpectralLLM Training")
    
    # Model parameters
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--harmonic_bases", type=int, default=16, help="Harmonic bases for spectral embedding")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    
    # Options
    parser.add_argument("--save_model", action="store_true", help="Save trained model")
    
    args = parser.parse_args()
    
    print("🌊 Simple SpectralLLM Training Example")
    print("=" * 50)
    print(f"Model: {args.embed_dim}d, {args.num_layers} layers, {args.num_heads} heads")
    print(f"Training: {args.epochs} epochs, batch size {args.batch_size}")
    print()
    
    train_spectralllm(args)


if __name__ == "__main__":
    main() 