"""
Enhanced Tokenizer for SpectralLLM
==================================

Simple but comprehensive tokenizer with character and word-level support.
"""

import json
import os
from typing import List, Dict, Optional
from collections import Counter


class SimpleTokenizer:
    """
    Enhanced character or word-level tokenizer with vocabulary management.
    """
    
    def __init__(self, mode: str = 'char', vocab_file: Optional[str] = None):
        """
        Initialize tokenizer.
        
        Args:
            mode: 'char' for character-level, 'word' for word-level
            vocab_file: Optional vocabulary file to load
        """
        self.mode = mode
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<bos>': 2,
            '<eos>': 3
        }
        
        # Initialize with special tokens
        self.vocab.update(self.special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_id = len(self.special_tokens)
        
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into units.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.mode == 'char':
            return list(text)
        else:  # word mode
            # Simple word tokenization (can be enhanced)
            return text.lower().split()
    
    def build_vocab(self, texts: List[str], max_vocab_size: int = 10000) -> None:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of training texts
            max_vocab_size: Maximum vocabulary size
        """
        print(f"🔤 Building {self.mode}-level vocabulary...")
        
        # Count all tokens
        token_counts = Counter()
        total_texts = len(texts)
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"  Processing text {i+1}/{total_texts}")
            
            tokens = self.tokenize(text)
            token_counts.update(tokens)
        
        print(f"  Found {len(token_counts)} unique tokens")
        
        # Keep most frequent tokens (excluding special tokens)
        available_slots = max_vocab_size - len(self.special_tokens)
        most_common = token_counts.most_common(available_slots)
        
        # Reset vocab to just special tokens
        self.vocab = self.special_tokens.copy()
        self.next_id = len(self.special_tokens)
        
        # Add most common tokens
        for token, count in most_common:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.next_id += 1
        
        # Update inverse mapping
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"✅ Vocabulary built: {len(self.vocab)} tokens")
        print(f"   Coverage: {sum(count for token, count in most_common if token in self.vocab)} / {sum(token_counts.values())} tokens")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        # Convert to IDs
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.special_tokens['<bos>'])
        
        for token in tokens:
            token_id = self.vocab.get(token, self.special_tokens['<unk>'])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<eos>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, '<unk>')
            
            # Skip special tokens if requested
            if skip_special_tokens and token in self.special_tokens:
                continue
                
            tokens.append(token)
        
        if self.mode == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None, 
                    padding: bool = True) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length (for truncation/padding)
            padding: Whether to pad sequences to same length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        all_token_ids = []
        
        for text in texts:
            token_ids = self.encode(text)
            
            # Truncate if needed
            if max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            all_token_ids.append(token_ids)
        
        # Padding if requested
        if padding and max_length:
            padded_ids = []
            attention_masks = []
            
            for token_ids in all_token_ids:
                # Pad to max_length
                attention_mask = [1] * len(token_ids)
                
                while len(token_ids) < max_length:
                    token_ids.append(self.special_tokens['<pad>'])
                    attention_mask.append(0)
                
                padded_ids.append(token_ids)
                attention_masks.append(attention_mask)
            
            return {
                'input_ids': padded_ids,
                'attention_mask': attention_masks
            }
        
        return {'input_ids': all_token_ids}
    
    def save_vocab(self, vocab_file: str) -> None:
        """
        Save vocabulary to file.
        
        Args:
            vocab_file: Path to save vocabulary
        """
        vocab_data = {
            'mode': self.mode,
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'vocab_size': len(self.vocab)
        }
        
        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Vocabulary saved to {vocab_file}")
    
    def load_vocab(self, vocab_file: str) -> None:
        """
        Load vocabulary from file.
        
        Args:
            vocab_file: Path to vocabulary file
        """
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.mode = vocab_data['mode']
        self.vocab = vocab_data['vocab']
        self.special_tokens = vocab_data['special_tokens']
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_id = len(self.vocab)
        
        print(f"📁 Vocabulary loaded from {vocab_file}")
        print(f"   Mode: {self.mode}, Size: {len(self.vocab)}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_vocab_stats(self) -> Dict:
        """Get vocabulary statistics."""
        return {
            'vocab_size': len(self.vocab),
            'mode': self.mode,
            'special_tokens': len(self.special_tokens),
            'regular_tokens': len(self.vocab) - len(self.special_tokens)
        } 