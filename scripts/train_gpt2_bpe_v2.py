#!/usr/bin/env python3
"""GPT-2 BPE Training on PLATO Ecosystem Data — v2 (smaller, faster)

Usage: python3 scripts/train_gpt2_bpe_v2.py --steps 3000 --batch 16

Uses tiktoken GPT-2 BPE tokenizer on collected workspace text.
Runs on CUDA if available.
"""
import torch, torch.nn as nn, tiktoken, glob, os, time, math, argparse

class Head(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h=h; self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d)
    def forward(self,x):
        B,T,C=x.shape; hs=C//self.h
        q,k,v=self.qkv(x).reshape(B,T,3,self.h,hs).permute(2,0,3,1,4)
        a=(q@k.transpose(-2,-1))/math.sqrt(hs)
        a=a.masked_fill(torch.triu(torch.ones(T,T,device=x.device),1).bool(),float('-inf'))
        return self.proj((torch.softmax(a,-1)@v).transpose(1,2).reshape(B,T,C))

class Block(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.l1=nn.LayerNorm(d); self.a=Head(d,h)
        self.l2=nn.LayerNorm(d); self.m=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x): x=x+self.a(self.l1(x)); return x+self.m(self.l2(x))

class GPT(nn.Module):
    def __init__(self,d=128,h=4,L=4,ctx=128,V=6168):
        super().__init__(); self.ctx=ctx
        self.emb=nn.Embedding(V,d); self.pos=nn.Embedding(ctx,d)
        self.blocks=nn.ModuleList([Block(d,h) for _ in range(L)])
        self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,V)
    def forward(self,x):
        B,T=x.shape; h=self.emb(x)+self.pos(torch.arange(T,device=x.device))
        for b in self.blocks: h=b(h)
        return self.head(self.ln(h))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ws = "/home/phoenix/.openclaw/workspace"
    texts = []
    for pattern in ["**/README.md", "**/MEMORY.md", "**/HEARTBEAT.md"]:
        for f in glob.glob(os.path.join(ws, pattern), recursive=True):
            try: texts.append(open(f).read())
            except: pass
    for f in glob.glob(os.path.join(ws, "signal-chain/papers/*.md")):
        try: texts.append(open(f).read())
        except: pass
    
    enc = tiktoken.get_encoding("gpt2")
    all_tokens = enc.encode("\n\n".join(texts))
    unique = sorted(set(all_tokens))
    tmap = {t:i for i,t in enumerate(unique)}
    V = len(unique)
    tokens = [tmap[t] for t in all_tokens]
    
    model = GPT(d=args.dim, h=4, L=4, ctx=args.ctx, V=V).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Vocab: {V} | Tokens: {len(tokens):,} | Params: {params:,} | Device: {device}")
    
    data = torch.tensor(tokens, dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    t0 = time.time()
    for step in range(args.steps):
        ix = torch.randint(len(data)-args.ctx, (args.batch,))
        x = torch.stack([data[i:i+args.ctx] for i in ix])
        y = torch.stack([data[i+1:i+args.ctx+1] for i in ix])
        loss = nn.functional.cross_entropy(model(x).view(-1,V), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"Step {step} | loss={loss.item():.3f} | ppl={math.exp(min(loss.item(),10)):.1f} | {time.time()-t0:.1f}s")
    
    model.eval()
    for p in ["The PLATO room", "Tiles are", "Constraint theory"]:
        ids = [tmap.get(t,0) for t in enc.encode(p)]
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            for _ in range(200):
                logits = model(x[:,-args.ctx:]); nxt = torch.softmax(logits[:,-1]/0.8,-1).multinomial(1)
                x = torch.cat([x,nxt],1)
        orig = [unique[i] for i in x[0].tolist()]
        print(f"\n'{p}': {enc.decode(orig)}")
    
    torch.save(model.state_dict(), "scripts/gpt2_bpe_checkpoint.pt")
    print(f"\nSaved checkpoint. Total: {time.time()-t0:.1f}s")
