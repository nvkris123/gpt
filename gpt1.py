# -*- coding: utf-8 -*-
"""Small GPT-style language model trained on Tiny Shakespeare."""

import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLM(nn.Module):
  def __init__(self, vocab_size, n_embd):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    print("embed table = ", self.token_embedding_table)

    self.lm_head = nn.Linear(n_embd, vocab_size)
    print("lm head = ", self.lm_head)

  def forward(self, idx, targets=None):
    tok_emb = self.token_embedding_table(idx)
    logits = self.lm_head(tok_emb)
    loss = None

    if targets is not None:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main():
  torch.manual_seed(1337)

  max_iters = 5000
  eval_interval = 500
  learning_rate = 1e-3
  eval_iters = 200
  batch_size = 32
  block_size = 8
  n_embd = 32
  device = "cpu"  # This small model is faster on CPU than MPS.

  file_path = os.path.join("tiny-shakespeare-input.txt")
  with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

  print("length of text=", len(text))
  chars = sorted(list(set(text)))
  print("chars=", chars)
  vocab_size = len(chars)
  print("vocab_size=", vocab_size)

  stoi = {ch: i for i, ch in enumerate(chars)}
  itos = {i: ch for i, ch in enumerate(chars)}
  encode = lambda s: [stoi[c] for c in s]
  decode = lambda l: "".join([itos[i] for i in l])

  #print(decode([33, 41]))
  print("device=", device)

  data = torch.tensor(encode(text), dtype=torch.long)
  print(data.shape, data.dtype)
  print(text[:10])
  print(data[:10])

  n = int(0.9 * len(data))
  train_data = data[:n]
  val_data = data[n:]

  x = train_data[:block_size]
  y = train_data[1:block_size+1]
  for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    # print(f"when input is {context} the target: {target}")
  # sys.exit()

  def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

  @torch.no_grad()
  def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
        X, Y = get_batch(split)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
      out[split] = losses.mean()
    model.train()
    return out

  xb, yb = get_batch("train")
  print(xb.shape, yb.shape)
  print(xb)
  print(yb)
  for b in range(batch_size):
    for t in range(block_size):
      context = xb[b, :t+1]
      target = yb[b, t]
      # print(f"when input is {context.tolist()} the target: {target}")

  model = BigramLM(vocab_size, n_embd).to(device)
  out, loss = model(xb, yb)
  print("out=", out.shape)
  print("loss=", loss)

  idx = torch.zeros((1, 1), dtype=torch.long, device=device)
  print(decode(model.generate(idx, max_new_tokens=100)[0].cpu().tolist()))

  print(model.parameters())
  print("-----")
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  for steps in range(max_iters):
    if steps % eval_interval == 0:
      losses = estimate_loss(model)
      print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  print(loss.item())

  idx = torch.zeros((1, 1), dtype=torch.long, device=device)
  print(decode(model.generate(idx, max_new_tokens=100)[0].cpu().tolist()))


if __name__ == "__main__":
  main()
