import torch
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class GPTConfig:
    # Arquitetura
    block_size: int = 512
    vocab_size: int | None = None
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: floaf = 0.1
    bias: bool = False

    # Config de Treino
    batch_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200

    # Otimizador e Taxa de aprendizado
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Caminhos e Diretórios
    data_path: str = str(PROJECT_ROOT / "data" / "raw" / "ascoo_dataset.txt")
    checkpoint: str = str(PROJECT_ROOT / "checkpoints")
    best_model_path: str = str(PROJECT_ROOT / "checkpoints" / "best_model.pt")
    last_model_path: str = str(PROJECT_ROOT / "checkpoints" / "last_model.pt")

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"