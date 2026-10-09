import os
import torch
import torch.nn.functional as F
from src.config import GPTConfig
from src.dataset import CharTokenizer
from src.model import MiniGPT
from src.utils import load_checkpoint

def generate():
    config = GPTConfig()
    device = config.device
    
    print(f"[*] Carregando tokenizer baseado no dataset: {config.data_path}")
    if not os.path.exists(config.data_path):
        print(f"[!] Erro: Dataset não encontrado em {config.data_path}")
        return
        
    with open(config.data_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    tokenizer = CharTokenizer(raw_text)
    config.vocab_size = tokenizer.vocab_size
    
    model = MiniGPT(config)
    model.to(device)
    
    checkpoint_path = config.best_model_path if os.path.exists(config.best_model_path) else config.last_model_path
    
   
    _, _ = load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    model.eval() 
    
    prompt = "\n"
    encoded_prompt = tokenizer.encode(prompt)
    

    if not encoded_prompt:
        encoded_prompt = [0]
        
    x = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0) # Shape: (1, T)
    
    max_new_tokens = 600  # Quantos caracteres o modelo vai gerar na tela
    temperature = 0.8     # Controla a criatividade (valores maiores = mais aleatório)
    top_k = 40            # Filtra apenas os top K caracteres mais prováveis
    
    print(f"\n[*] Gerando saída com o modelo (Prompt inicial: {repr(prompt)}):\n" + "="*50)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = x[:, -config.block_size:]
            
            logits, _ = model(idx_cond)
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, idx_next), dim=1)
            
            char_gerado = tokenizer.decode(idx_next[0].tolist())
            print(char_gerado, end='', flush=True)
            
    print("\n" + "="*50 + "\n[*] Geração concluída.")

if __name__ == "__main__":
    generate()