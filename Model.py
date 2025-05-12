# chess_model_reworked.py
# Полная версия модели с конволюциями → dense → self‑attention,
# где token_in входит в трансформер как второй токен, а token_out берётся из seq[:,1,:]

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

device = torch.device(
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

# ------------------------------------------------------------------
# 🔸 Inception‑подобный много‑масштабный блок (3×3, 5×5, 7×7) без BN
# ------------------------------------------------------------------
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels: int,
                 c3: int = 64, c5: int = 64, c7: int = 64):
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, c5, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5, c5, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, c7, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c7, c7, 7, padding=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch3(x),
            self.branch5(x),
            self.branch7(x)
        ], dim=1)  # (B, 192, 8, 8)


# ------------------------------------------------------------------
# 🔸 Простой residual блок 3×3 → 3×3 без BN
# ------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.dropout(self.conv2(out))
        return x + out  # skip‑connection


# ------------------------------------------------------------------
# 🔸 ChessModel
# ------------------------------------------------------------------
class ChessModel(nn.Module):
    """
    Принимает:
        board_x  – (B,65,16)     тензор состояния
        token_in – (B,512)       скрытый токен от предыдущего хода
        mask     – (B,4096) bool маска легальных ходов
    Возвращает:
        logits     – (B,4096)
        token_out  – (B,512)     обновлённый скрытый токен
    """
    def __init__(self,
                 in_channels: int = 256,
                 token_dim: int = 512,
                 n_heads: int = 16,
                 n_attn_layers: int = 6,
                 n_res_blocks: int = 4,
                 dropout: float = 0.):
        super().__init__()
        self.token_dim = token_dim
        self.dense_dim = token_dim // 4
        self.conv_dim  = 3 * token_dim // 4

        # 1) conv‑ветка + MultiScale + residual tower
        self.stem = nn.Sequential(
            nn.Conv2d(16, in_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ms = MultiScaleConvBlock(
            in_channels, self.dense_dim, self.dense_dim, self.dense_dim
        )
        self.squeeze   = nn.Conv2d(self.conv_dim, self.conv_dim, 1)
        self.res_tower = nn.Sequential(
            *[ResidualBlock(self.conv_dim, dropout) for _ in range(n_res_blocks)]
        )

        # 2) dense‑ветка для каждой клетки
        self.cell_dense = nn.Sequential(
            nn.Linear(16, self.dense_dim), nn.ReLU(inplace=True),
            nn.Linear(self.dense_dim, self.dense_dim), nn.ReLU(inplace=True)
        )

        # 3) спец‑токен из информации о поле
        self.special_embed = nn.Linear(16, token_dim)

        # 4) Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=4*token_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_attn_layers)

        # 5) политика и value‑MLP
        self.repr_mlp = nn.Sequential(
            nn.Linear(token_dim, 2*token_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2*token_dim, token_dim), nn.ReLU(inplace=True)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(token_dim, 1024), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 4096)
        )

        self.value_head = nn.Sequential(
            nn.Linear(token_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1)
        )


    def representation(self,
                       x: torch.Tensor,
                       token_in: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x        – (B,65,16)
        token_in – (B,512)
        Возвращает:
          h         – (B,token_dim) для политики
          token_out – (B,token_dim) обновлённый токен из трансформера
        """
        B = x.size(0)
        special, cells = x[:, 0, :], x[:, 1:, :]  # (B,16) и (B,64,16)

        # 1) свёртки → карты признаков
        feat = cells.view(B, 8, 8, 16).permute(0, 3, 1, 2)  # (B,16,8,8)
        y = self.squeeze(self.ms(self.stem(feat)))         # (B,conv_dim,8,8)
        y = self.res_tower(y)                              # (B,conv_dim,8,8)
        y = y.view(B, self.conv_dim, 64).transpose(1, 2)   # (B,64,conv_dim)

        # 2) dense‑признаки клеток
        d = self.cell_dense(cells)                         # (B,64,dense_dim)

        # 3) объединяем в токены клеток
        tokens_cells = torch.cat([y, d], dim=-1)           # (B,64,token_dim)

        # 4) готовим вход для трансформера
        cls_tok  = self.special_embed(special).unsqueeze(1)   # (B,1,token_dim)
        in_tok   = token_in.unsqueeze(1)                      # (B,1,token_dim)
        # print("in_tok: ", in_tok.shape)
        seq      = torch.cat([cls_tok, in_tok, tokens_cells], dim=1)
        #          └──  index=0   index=1      index=2..65   ──┘

        # 5) пропускаем через Transformer
        seq = self.transformer(seq)  # (B,66,token_dim)

        # 6) вынимаем выходы
        h         = self.repr_mlp(seq[:, 0, :])  # из CLS
        token_out = seq[:, 1, :]                 # обновлённый скрытый токен
        # print("token_out: ", token_out.shape)
        return h, token_out

    def forward(self,
                board_x: torch.Tensor,
                token_in: torch.Tensor,
                mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        board_x  – (B,65,16)
        token_in – (B,512)
        mask     – (B,4096)
        """
        h, token_out = self.representation(board_x, token_in)
        logits = self.policy_head(h)
        value = self.value_head(token_out).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        return logits, token_out, value


# ------------------------------------------------------------------
# 🔸 Утилиты
# ------------------------------------------------------------------
def index_to_move(idx: int) -> chess.Move:
    return chess.Move(idx // 64, idx % 64)

def evaluate_move(board: chess.Board, mv: chess.Move) -> bool:
    return mv in board.legal_moves


# ------------------------------------------------------------------
# 🔸 Быстрый тест
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = ChessModel().to(device)
    dummy_x     = torch.zeros((2, 65, 16), device=device)
    dummy_token = torch.zeros((2, 512), device=device)
    logits, tok = model(dummy_x, dummy_token)
    print("logits shape:", logits.shape)    # (2,4096)
    print("token_out shape:", tok.shape)    # (2,512)
    print("≈ # params:", sum(p.numel() for p in model.parameters()))
