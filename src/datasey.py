import torch
import os
from typing import Tuple
from scr.config import GPTConfig

class charTokenizer:
    "Tokenizador Character-Level"
    def __init__(self, data: str):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, string: str) -> list[int]:
        return[self.stoi[c] for c in string if c in self.stoi]
    
    def decode(self, indices: list[int]) -> str:
        return ''.join([self.itos[i] for i in indices])

class asciiDataset:
    def __init__(self, config: GPTConfig):
        self.config = config
        
        if not os.path.exists(config.data_path):
            raise FileNotFoundError(f"Erro: O arquivo de dataset não foi encontrado em {config.data_path}")

        with open(config.data_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        if len(raw_text) == 0:
            raise ValueError(f"O arquivo {config.data_path} está vazio. Adicione suas artes ASCII lá.")

        self.tokenizer = charTokenizer(raw_text)
        self.config.vocab_size = self.tokenizer.vocab_size

        print(f"Dataset carregado: {len(raw_text):,} caracteres")
        print(f"Tamanho do vocalulario (caracteres unicos): {self.vocab_size}")

        data_tensor = torch.tensor(self.tokenizer.encode(raw_text), dtype=torch.long)

        n = int(0.9 * len(data_tensor))
        self.train_data = data_tensor[:n]
        self.val_data = data_tensor[n:]

    def get_batch(self, split: str):
        data = self.train_data if split == 'train' else self.val_data
        
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        
        x = torch.stack([data[i : i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.config.block_size + 1] for i in ix])
        
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y