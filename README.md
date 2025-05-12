# ♟️ AI\_chess\_player

A Proximal Policy Optimization (PPO)–Transformer agent trained to play chess from scratch using reinforcement learning.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/SargisVardanian/AI_chess_player.git
cd AI_chess_player

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install python-chess

# 3. Run a self-play game
python game.py
```

Upon first execution, `game.py` will automatically download the pretrained model weights from GitHub Releases.

---

## Pretrained Weights

| Filename                                   | Size   | SHA-256 Digest |
| ------------------------------------------ | ------ | -------------- |
| `chess_model_transformer_weights_exp2.pth` | 157 MB | `ed20e8b5…`    |

The weights are fetched at runtime from:

```
https://github.com/SargisVardanian/AI_chess_player/releases/download/v1.0-weights/chess_model_transformer_weights_exp2.pth
```

---

## Repository Structure

```
AI_chess_player/
├─ game.py                   # Single-game simulator + auto-weight download
├─ train_ppo_value.py        # PPO training loop with value critic
├─ Chess_background.py       # Environment, reward shaping, and utilities
├─ chess_model_reworked.py   # Transformer–CNN hybrid network
├─ images/                   # Chess piece sprite assets
├─ .github/
│   └─ workflows/
│       └─ run_game.yml      # “play-demo” GitHub Actions workflow
├─ .gitattributes            # Git LFS configuration
├─ .gitignore
└─ README.md
```

---

## Technical Overview

### PPO Training (`train_ppo_value.py`)

We optimize a policy π\_θ(a|s) via the clipped surrogate objective \[1]:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \big[ \min(r_t(\theta) \hat{A}_t,\; \mathrm{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\,\hat{A}_t) \big],
$$

where
$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$
and the advantage estimate $\hat{A}_t$ uses one-step TD bootstrap with discount $\gamma=0.99$.

The total loss includes a value-function term and entropy regularization:

$$
L(\theta) = L^{\text{CLIP}}(\theta) + c_{v}\,L^{V}(\theta)\;-\;c_{e}\,H[\pi_\theta](s_t),
$$

with $c_v=0.9$, $c_e=0.02$, and Kullback–Leibler penalty weight $\beta=0.005$.

Each epoch iterates over a batch of 64 parallel `ChessGame` environments, applying gradient updates every step $t$. Models are checkpointed every 50 steps.

---

### Model Architecture (`chess_model_reworked.py`)

The agent’s backbone is a hybrid CNN–Transformer encoder:

1. **Multi-Scale Convolutional Block**
   Three parallel convolutions (kernel sizes 3×3, 5×5, 7×7) concatenated and followed by a residual tower (4 × 3×3 residual blocks).

2. **Cell-Wise Dense Embedding**
   A two-layer MLP mapping each of the 64 squares’ feature vectors into a $d_{\text{dense}}$-dimensional embedding.

3. **Special CLS & Memory Token**

   * A special \[CLS] token embedding global board context.
   * A learnable `token_in` (dim 512) carries temporal memory between moves.

4. **Transformer Encoder**  
   Six layers ($d_{\text{model}} = 512,\; h = 16$) process the sequence  
   `[CLS]`, `token_in`, and `cells_{1..64}` concatenated as  
   `\bigl[[CLS], token_in, cells_{1..64}\bigr]`.


5. **Heads**

   * **Policy:** MLP → 4096 logits (all possible moves) with legal-move masking.
   * **Value:** MLP → scalar $V(s)$.

Total learnable parameters: ≈ 30 million.

---

## Interactive Execution on GitHub

### Codespaces

Launch an interactive development environment in seconds:

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?repo=SargisVardanian/AI_chess_player)

### GitHub Actions Demo

1. Go to **Actions → play-demo**.
2. Click **Run workflow**.
3. Inspect the logs for a step-by-step record of the agent’s moves.

The workflow `.github/workflows/run_game.yml` sets up Python 3.10, installs dependencies, and executes `python game.py`.

---

## References

\[1] Schulman, J. et al. “Proximal Policy Optimization Algorithms.” arXiv:1707.06347 (2017).
