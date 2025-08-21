import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import pickle
from pathlib import Path

class TextDataset(Dataset):
    def __init__(self, tokens, max_length, stride=None, pad_token_id=0):
        self.tokens = tokens
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length
        self.pad_token_id = pad_token_id

    def __len__(self):
        return max(1, (len(self.tokens) - self.max_length) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = min(start + self.max_length, len(self.tokens))
        chunk = self.tokens[start:end]
        if len(chunk) < self.max_length:
            chunk = chunk + [self.pad_token_id] * (self.max_length - len(chunk))
        return torch.tensor(chunk, dtype=torch.long)

class SharedDataManager:
    """Manages shared tokenized data across multiple experiment runs"""
    
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.config = None
        
    def load_or_create_datasets(self, config, force_reload=False):
        """Load cached datasets or create new ones"""
        cache_file = self.cache_dir / f"datasets_{config.num_documents}_{config.max_seq_len}.pkl"
        
        if not force_reload and cache_file.exists():
            print("🔄 Loading cached datasets...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.tokenizer = cached_data['tokenizer']
                    self.train_dataset = cached_data['train_dataset']
                    self.val_dataset = cached_data['val_dataset']
                    self.config = cached_data['config']
                print("✅ Cached datasets loaded successfully!")
                return self._create_dataloaders(config)
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}")
                print("🔄 Creating new datasets...")
        
        print("🔄 Creating new datasets...")
        return self._create_new_datasets(config, cache_file)
    
    def _create_new_datasets(self, config, cache_file):
        """Create new tokenized datasets"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        config.pad_token_id = self.tokenizer.pad_token_id
        config.vocab_size = self.tokenizer.vocab_size
        
        # Load dataset
        dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 
                              split="train", streaming=True)
        
        # Tokenize documents
        print("📝 Tokenizing documents...")
        all_documents = []
        for i, item in enumerate(tqdm(dataset, total=config.num_documents, desc="Tokenizing")):
            if i >= config.num_documents:
                break
            tokens = self.tokenizer.encode(item["text"][:4000], add_special_tokens=False)
            all_documents.append(tokens)
        
        # Train/val split
        n_train = int(len(all_documents) * 0.85)
        train_docs = all_documents[:n_train]
        val_docs = all_documents[n_train:]
        
        # Flatten
        train_tokens = [token for doc in train_docs for token in doc]
        val_tokens = [token for doc in val_docs for token in doc]
        
        # Create datasets
        self.train_dataset = TextDataset(
            train_tokens, 
            config.max_seq_len, 
            stride=int(config.max_seq_len * 0.8),
            pad_token_id=config.pad_token_id
        )
        self.val_dataset = TextDataset(
            val_tokens, 
            config.max_seq_len, 
            stride=config.max_seq_len,
            pad_token_id=config.pad_token_id
        )
        
        print(f"📚 Data: {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens")
        print(f"📊 Data: {len(self.train_dataset)} train sequences, {len(self.val_dataset)} val sequences")
        
        # Cache the datasets
        self._cache_datasets(cache_file, config)
        
        return self._create_dataloaders(config)
    
    def _create_dataloaders(self, config):
        """Create DataLoaders from existing datasets"""
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config.batch_size,
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=config.batch_size,
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _cache_datasets(self, cache_file, config):
        """Cache the created datasets"""
        try:
            cache_data = {
                'tokenizer': self.tokenizer,
                'train_dataset': self.train_dataset,
                'val_dataset': self.val_dataset,
                'config': config
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Datasets cached to {cache_file}")
        except Exception as e:
            print(f"⚠️ Failed to cache datasets: {e}")
    
    def get_tokenizer(self):
        """Get the loaded tokenizer"""
        return self.tokenizer
    
    def clear_cache(self):
        """Clear all cached datasets"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print("🗑️ Cache cleared")

# Global instance
shared_data_manager = SharedDataManager()
