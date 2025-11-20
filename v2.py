import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# ----hyperparameters----------------------------------
batch_size = 32
block_size = 8
iterations = 3000
learning_rate = 1e-2
train_test_split = 0.9 # 90% train and 10% validation
eval_interval = 300
eval_iters = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embed = 32
# -----------------------------------------------------

path = './data/input.txt'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# create the vocabulary
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# create a basic character level tokenizer
stoi = { ch: i for i, ch in enumerate(vocab)}
itos = { i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda n: ''.join([itos[i] for i in n])

# split data to train and validation set
data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_test_split * len(text))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLamguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, context, targets=None):
        # context and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(context) # (B, T, C)
        logits = self.lm_head(tok_emb) # (B, T, vocab_size)


        # calculate the loss if targets are given
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context, max_new_tokens):
        # context is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(context)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            context = torch.cat((context, next_token), dim=1)  # (B, T+1)
        return context


model = BigramLamguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(iterations):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context=context, max_new_tokens=500)[0].tolist()))
