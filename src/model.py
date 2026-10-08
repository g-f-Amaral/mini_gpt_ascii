import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.config import GPTConfig

class causalSelfAttention(nn.Module):
    def __init__(self,config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "A dimensão de embedding deve ser divisível pelo número de cabeças"

        #Key, Query, Value
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Saida
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_droupout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        mask = torch.trill(torch.ones(config.block_size, config.batch_size))
        self.register_buffer("bias", mask.view(1,1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # Batch, Tempo, Canal (n_embd)

        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k ,v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_droupout(self.c_proj(y))
        return y

class feedForward(nn.Module):
    "Rede MLP simples com SiLU"
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act = nn.SiLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class block(nn.Module):
    "self attention + feed forward"
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = causalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = feedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class miniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None, "O vocab_size deve ser inicializado antes de criar o modelo!"
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([block(config) for _ in range(config.n_layer)]),
            ln_f = nn.RMSNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(modulem, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        def foward(self, idx, target=None):
            device = idx.device
            B, T = idx.size()
            assert T <= self.config.block_size, f"Não é possível processar sequência {T}, max={self.config.block_size}"

            pos = torch.arange(0, T, dtype=torch.long, device=device)

            tok_emb = self.transformer.wte(idx) # B, T, C
            pos_emb = self.transformer.wpe(pos) # T, C

            x = self.transformer.drop(tok_emb + pos_emb)

            for block in self.transformer.h:
                x = block(x)

            x = self.transformer.ln_f(x)

            if targets is not None:
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits(-1)), targets.view(-1), ignore_index=-1)
            else:
                logits =self.lm_head(x[:, -1, :])
                return logits, None
