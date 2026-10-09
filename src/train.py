import os
import torch
import torch.optim as optim
from tqdm import tqdm 

from src.config import GPTConfig
from src.dataset import ASCIIDataset
from src.model import MiniGPT
from src.utils import save_checkpoint, load_checkpoint


# Validation Loss

@torch.no_grad()
def estimate_loss(model, dataset, eval_iters, device):
    out = {}
    model.eval() 
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataset.get_batch(split)
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
        
    model.train() 
    return out

# Treinamento

def train():
    config = GPTConfig()
    device = config.device
    print(f"[*] Iniciando processo de treinamento usando: {device.upper()}")

    dataset = ASCIIDataset(config)
    
    model = MiniGPT(config)
    model.to(device)

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Instancia AdamW
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )


    # Tenta retomar de um Checkpoint antigo se ele existir
    start_iter, best_val_loss = 0, float('inf')
    if os.path.exists(config.last_model_path):
        print(f"[*] Tentando carregar checkpoint recente em {config.last_model_path}...")
        start_iter, best_val_loss = load_checkpoint(config.last_model_path, model, optimizer, device)

   
    # Loop de Treinamento 
    print("\n[*] Iniciando o Loop de Treino. Pressione Ctrl+C para parar com segurança a qualquer momento.")
    
    progress_bar = tqdm(range(start_iter, config.max_iters), initial=start_iter, total=config.max_iters)

    try:
        for iter_num in progress_bar:
            
            if iter_num > 0 and iter_num % config.eval_interval == 0:
                losses = estimate_loss(model, dataset, config.eval_iters, device)
                
                print(f"\n[Iter {iter_num}] loss treino: {losses['train']:.4f} | loss valid: {losses['val']:.4f}")
                
                is_best = losses['val'] < best_val_loss
                if is_best:
                    best_val_loss = losses['val']
                    
                save_checkpoint(model, optimizer, iter_num, best_val_loss, config, is_best)

            X, Y = dataset.get_batch('train')

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=use_amp):
                logits, loss = model(X, targets=Y)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            progress_bar.set_description(f"Loss: {loss.item():.4f}")

    except KeyboardInterrupt:
        # Se você apertar Ctrl+C no terminal, ele não corrompe o treino, ele intercepta e salva a parada
        print("\n[!] Treinamento interrompido pelo usuário. Salvando último estado...")
        save_checkpoint(model, optimizer, iter_num, best_val_loss, config, is_best=False)

    print("\n[*] Fim do processo.")

if __name__ == '__main__':
    # Para garantir compatibilidade de memória no Windows com PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train()