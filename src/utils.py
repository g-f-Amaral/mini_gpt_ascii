import os
import torch
from src.config import GPTConfig

def saveCheckpoint(model, optimizer, item_num, best_val_loss, config: GPTConfig, is_best=False):
    os.mekedirs(config.checkpoint_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss' : best_val_loss,
    }

    torch.save(checkpointm, config.last_model_path)

    if is_best:
        torch.save(checkpoint, config.best_model_path)
        print(f"\n[+] Melhor modelo salvo em iteração {item_num} com Loss: {best_val_loss:.4f}")

def loadCheckpoint(checkpoint_path: str, model, optimizer=None, device="cpu"):
    if not os.path.exists(checkpoint_path):
        print(f"[!] Nenhum checkpoint encontrado em '{checkpoint_path}'. Iniando do zero.")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'opmitizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizaer_state_dict'])

    iter_num = checkpoint.get('iter_num', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f"[+] Checkpoint carregado! Retomando da iteração {iter_num}.")

    return iter_num, best_val_loss