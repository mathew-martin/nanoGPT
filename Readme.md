nanoGPT mini-experiments
========================

Lightweight character-level language models inspired by Andrej Karpathy's nanoGPT tutorials. The repo includes a simple bigram baseline and a small transformer-style model you can train on the bundled text corpus or your own data.

Repo layout
-----------
- `bigram.py` – bare-bones bigram language model that learns next-character logits directly from embeddings.
- `gpt.py` – small GPT-style model with token + positional embeddings, multi-head self-attention, and feed-forward blocks.
- `data/input.txt` – default training corpus; replace with your own text to train on a different domain.
- `gpt.ipynb` – notebook scratchpad for experimentation.

Setup
-----
1) Python 3.10+ recommended. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2) Install dependencies (PyTorch is the only requirement for the scripts):
```bash
pip install torch
```
If you need CUDA wheels, see the install selector at https://pytorch.org for the correct command.

Running the models
------------------
- Train and sample from the bigram baseline:
```bash
python bigram.py
```
- Train and sample from the transformer model:
```bash
python gpt.py
```
Both scripts will print periodic training/validation losses and then generate 500 characters of sample text. They automatically use a GPU when `torch.cuda.is_available()` is true, otherwise they run on CPU.

Customization tips
------------------
- Swap `data/input.txt` with any UTF-8 text file to train on new data.
- Tweak hyperparameters near the top of each script (`block_size`, `iterations`, `learning_rate`, etc.) to change context length, training duration, and optimization settings.
- Generation length is controlled by `max_new_tokens` in each script's final call to `model.generate`.
