# train_ppo_value.py
import os, csv, numpy as np
import torch, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Categorical

from Model import ChessModel            # как в вашем chess_model_reworked.py
from Chess_background import ChessGame, states_board_and_masks

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ───────── Hyperparameters ─────────
CLIP_EPS   = 0.2
KL_BETA    = 0.005
ENT_COEF   = 0.02
VF_COEF    = 0.9        # вес критика
GAMMA      = 0.99
PENALTY_K  = 0.8
SAVE_INT   = 50
BATCH = 64

def count_parameters(model: torch.nn.Module):
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {tot:,},  trainable: {trn:,}")

# ─────────────────────────────────────────────────────────────────────
def train(model: ChessModel,
          optimizer: optim.Optimizer,
          ckpt_path: str,
          log_csv: str,
          epochs: int = 200_000):

    model.to(device).train()
    count_parameters(model)

    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['Ep','Step','Loss','PLoss','VLoss','KL','Ent','R̄_prev']
        )
    for ep in range(1, epochs+1):
        envs = [ChessGame(i) for i in range(BATCH)]
        states, _, masks = states_board_and_masks(envs, device=device)
        tokens = torch.zeros(BATCH, model.token_dim, device=device)

        # t‑1 буферы
        ps = pm = pa = pl = pr = pv = pt = None
        done = [False]*BATCH
        step = 0

        while not all(done):

            # ===== 1. шаг t  ==================================================
            logits_t, tokens_next, values_t = model(states, tokens, masks)
            dist_t = Categorical(F.softmax(logits_t, -1))
            actions_t = dist_t.sample()
            logps_t = dist_t.log_prob(actions_t)

            envs, raw_r, done, *_ , kcaps, _ = \
                ChessGame.process_moves(envs, actions_t.tolist())

            r_t = torch.as_tensor(raw_r,  dtype=torch.float32, device=device)
            kmask = torch.as_tensor(kcaps, dtype=torch.bool,   device=device)

            # ===== 2. PPO‑update на пакете (t‑1) ==============================
            if ps is not None:                                     # (данные t‑1)
                # --- returns & advantages ------------------------------------
                shaped_r = pr.clone()
                shaped_r[kmask] -= PENALTY_K * r_t[kmask]

                done_t1  = torch.as_tensor(done, device=device)
                # TD‑цель для t‑1: r_{t‑1}+γ·V(s_t)              (V(s_t) ≈ values_t)
                v_target_prev = shaped_r + GAMMA*values_t.detach()* (~done_t1)

                adv = (v_target_prev - pv.detach())
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

                # --- policy loss (t‑1) ---------------------------------------
                logits_prev, _, _ = model(ps, pt, pm)            # pt = prev_tokens
                dist_prev = Categorical(F.softmax(logits_prev, -1))
                logp_prev = dist_prev.log_prob(pa)       # pa,pl – t‑1

                ratio = torch.exp(logp_prev - pl)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1+CLIP_EPS) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- critic loss (t, текущее состояние!) ---------------------
                # r_t + γ·V(s_{t+1})  появится только на след. шаге,
                # поэтому берём one‑step TD‑return с bootstrap‑ом = values_t
                v_target_current = r_t + GAMMA*values_t.detach()* (~done_t1)
                value_loss = 0.5*F.mse_loss(values_t.squeeze(-1), v_target_current)

                # --- доп. метрики --------------------------------------------
                entropy = dist_prev.entropy().mean()
                kl = (pl - logp_prev).mean()

                loss = policy_loss + VF_COEF*value_loss \
                                   + KL_BETA*kl           \
                                   - ENT_COEF*entropy

                opt.zero_grad()
                loss.backward()
                opt.step()

                # лог
                m_r = shaped_r.mean().item()
                print(f"[Ep{ep}] step {step:>4} | "
                      f"L={loss:+.4f} PL={policy_loss:+.4f} "
                      f"VL={value_loss:+.4f}, {values_t.mean().item():+.4f} KL={kl:+.4f} "
                      f"Ent={entropy:+.4f} R̄_prev={m_r:+.3f}")

            # ===== 3. shift t → t‑1  =========================================
            ps, pm  = states, masks
            pa, pl  = actions_t.detach(), logps_t.detach()
            pr, pv  = r_t.detach(), values_t.detach()
            pt      = tokens.detach()

            # ===== 4. prepare следующий цикл ================================
            tokens  = tokens_next.detach()
            if any(done):
                tokens[torch.as_tensor(done, device=device)] = 0.

            states, _, masks = states_board_and_masks(envs, device=device)
            step += 1

            if step % SAVE_INT == 0:
                torch.save(model.state_dict(), ckpt_path)
                print(f"[Checkpoint] saved at step {step}")

        torch.save(model.state_dict(), ckpt_path)
        print(f"[Epoch {ep} done, checkpoint saved]")

    print("Training finished.")

# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    CKPT  = 'chess_model_transformer_weights_exp2.pth'
    LOG   = 'grpo_stepwise_log.csv'
    LR    = 1e-4

    model = ChessModel().to(device)
    if os.path.exists(CKPT):
        try:
            model.load_state_dict(torch.load(CKPT, map_location=device),
                                  strict=False)
            print("Model loaded.")
        except Exception as e:
            print("Load error:", e)
    else:
        print("No checkpoint found — training from scratch.")

    opt = optim.Adam(model.parameters(), lr=LR)
    train(model, opt, CKPT, LOG)
