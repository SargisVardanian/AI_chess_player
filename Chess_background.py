import chess, torch

# ──────────────────────────────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def board_to_matrix(board):
    squares_int   = torch.arange(64, device=device).view(8, 8)
    squares       = squares_int.float() / 63.0
    color_matrix  = ((squares_int % 2) + ((squares_int // 8) % 2)) % 2
    color_matrix  = color_matrix.float()

    pieces = torch.tensor(
        [board.piece_type_at(sq) or 0 for sq in chess.SQUARES],
        dtype=torch.int8, device=device).view(8, 8)
    piece_colors = torch.tensor(
        [1 if board.color_at(sq) == chess.WHITE else
         -1 if board.color_at(sq) == chess.BLACK else 0
         for sq in chess.SQUARES],
        dtype=torch.int8, device=device).view(8, 8)

    M = torch.zeros((8, 8, 4), dtype=torch.float32, device=device)
    M[..., 0] = squares
    M[..., 1] = color_matrix
    M[..., 2] = piece_colors.float()
    M[..., 3] = pieces.float()
    return M

def add_turn_vector(board_matrix, current_turn):
    flat = board_matrix.view(64, 4)
    turn_vec       = torch.zeros((1, 4), dtype=torch.float32, device=device)
    turn_vec[0, 0] = current_turn
    return torch.cat([turn_vec, flat], dim=0)

# ──────────────────────────────────────────────────────────────────────────
def index_to_move(index): return chess.Move(index // 64, index % 64)
def move_to_index(move):  return move.from_square * 64 + move.to_square
def move_key(move):       return (min(move.from_square, move.to_square),
                                  max(move.from_square, move.to_square))

piece_values = {
    chess.PAWN: 0.0, chess.KNIGHT: 0., chess.BISHOP: 0.,
    chess.ROOK: 0., chess.QUEEN: 4., chess.KING: 0.
}

# ──────────────────────────────────────────────────────────────────────────
class ChessGame:
    """
    Шахматная среда c двумя токенами скрытого состояния:
        hidden_tokens[chess.WHITE]  – токен белых
        hidden_tokens[chess.BLACK]  – токен чёрных
    На каждый ход в модель подаётся токен текущей стороны,
    а полученный `token_next` сохраняется обратно в ту же ячейку.
    """
    def __init__(self, game_id: int = 0):
        self.game_id = game_id
        self.reset()                                       # <‑‑ создаёт все поля
        self.state_buffer_white, self.state_buffer_black = [], []

    # ────────────────────────────────────────────────────────────────────
    def reset(self):
        self.board            = chess.Board()
        self.game_over        = False
        self.game_over_reason = ""
        self.last_move_was_legal = True
        self.agent_won        = None

        # ⇦ NEW  два токена
        self.hidden_tokens = {
            chess.WHITE: torch.zeros(512, device=device),
            chess.BLACK: torch.zeros(512, device=device)
        }

        self.move_history_white, self.move_history_black = [], []
        self.white_king_sq = self.board.king(chess.WHITE)
        self.black_king_sq = self.board.king(chess.BLACK)

        self.captured_piece, self.moved_king, self.did_castle = None, False, False
        self.last_move_key_white = self.last_move_key_black = None
        self.repeated_count_white = self.repeated_count_black = 0
        self.repeat_penalty_white = self.repeat_penalty_black = 0.0
        self.state_buffer_white, self.state_buffer_black = [], []
        self.king_missed = False

        self.current_agent_color = chess.WHITE
        self.color_str           = "WHITE"

    # ⇦ NEW — удобные геттер/сеттеры -------------------------------
    def get_current_token(self) -> torch.Tensor:
        return self.hidden_tokens[self.current_agent_color]

    def set_current_token(self, token: torch.Tensor):
        self.hidden_tokens[self.current_agent_color] = token.detach()

    # ────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        reward, done, illegal = self.play_move(action)
        next_state            = self.update()
        return next_state, reward, done, {"illegal": illegal}

    # ────────────────────────────────────────────────────────────────────
    def update(self):
        current_turn_flag = 1 if self.current_agent_color == chess.WHITE else 0
        mat   = board_to_matrix(self.board)
        state = add_turn_vector(mat, current_turn_flag)            # [65,4]

        buf   = self.state_buffer_white if self.current_agent_color == chess.WHITE \
                else self.state_buffer_black
        buf.append(state.clone())
        while len(buf) < 4: buf.insert(0, state.clone())
        buf = buf[-4:]
        if self.current_agent_color == chess.WHITE:
            self.state_buffer_white = buf
        else:
            self.state_buffer_black = buf

        return torch.cat(buf, dim=-1)

    def _update_move_history(self, move, moved_color, moved_color_str):
        mk = move_key(move)
        if moved_color == chess.WHITE:
            if self.last_move_key_white == mk:
                self.repeated_count_white += 1
                if self.repeated_count_white >= 2:
                    self.repeat_penalty_white = max(-1.0 * self.repeated_count_white, -3.0)
            else:
                self.last_move_key_white = mk
                self.repeated_count_white = 1
                self.repeat_penalty_white = 0.0
        else:
            if self.last_move_key_black == mk:
                self.repeated_count_black += 1
                if self.repeated_count_black >= 2:
                    self.repeat_penalty_black = max(-1.0 * self.repeated_count_black, -3.0)
            else:
                self.last_move_key_black = mk
                self.repeated_count_black = 1
                self.repeat_penalty_black = 0.0

    def play_move(self, action):
        """
        Основная логика:
          1) Смотрим, не было ли хода на взятие короля (до push).
          2) Если такой ход был, но мы не выбрали именно его, => king_missed = True
          3) Выполняем реальный ход (action).
          4) Возвращаем (reward, done, illegal).
        """
        if self.game_over:
            self.reset()
            return 0.0, True, False

        self.king_missed = False  # Сбрасываем каждый раз перед ходом

        moved_color = self.current_agent_color
        moved_color_str = "WHITE" if moved_color == chess.WHITE else "BLACK"

        board_before = self.board.copy(stack=False)
        pseudo_legal_before = list(board_before.generate_pseudo_legal_moves())

        # Проверяем, есть ли ход, бьющий короля
        # (упрощённо: если на to_square стоит король противника)
        can_capture_king = False
        king_capture_moves = []
        for mv in pseudo_legal_before:
            piece_captured = board_before.piece_at(mv.to_square)
            if piece_captured and piece_captured.piece_type == chess.KING:
                if piece_captured.color != moved_color:
                    # Это ход, которым можно бить короля
                    can_capture_king = True
                    king_capture_moves.append(mv)

        move = index_to_move(action)
        piece = self.board.piece_at(move.from_square)

        # Проверяем, выбрали ли мы именно ход на взятие короля
        # Если can_capture_king == True, но move не в king_capture_moves, => упущено
        if can_capture_king and move not in king_capture_moves:
            self.king_missed = True

        # -- Стандартные проверки легальности --
        if piece is None or piece.color != moved_color:
            self.last_move_was_legal = False
            self.game_over = True
            self.agent_won = False
            self.game_over_reason = "illegal_move_wrong_color"
            reward = self.get_reward(moved_color, moved_color_str)
            return reward, True, True

        if piece.piece_type == chess.PAWN:
            target_rank = chess.square_rank(move.to_square)
            if (piece.color == chess.WHITE and target_rank == 7) or \
               (piece.color == chess.BLACK and target_rank == 0):
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

        pseudo_legal = list(self.board.generate_pseudo_legal_moves())
        if move not in pseudo_legal:
            self.last_move_was_legal = False
            self.game_over = True
            self.agent_won = False
            self.game_over_reason = "illegal_move"
            reward = self.get_reward(moved_color, moved_color_str)
            return reward, True, True

        if piece.piece_type == chess.KING and piece.color == moved_color:
            self.moved_king = True
            if self.board.is_castling(move):
                self.did_castle = True

        captured_piece = self.board.piece_at(move.to_square)
        if captured_piece:
            self.captured_piece = captured_piece

        # Делаем ход
        self.board.push(move)
        self._update_move_history(move, moved_color, moved_color_str)

        # Проверяем, не убили ли короля
        if self.captured_piece and self.captured_piece.piece_type == chess.KING:
            self.game_over = True
            if self.captured_piece.color != moved_color:
                self.agent_won = True
                self.game_over_reason = "opponent_king_captured"
                print(f"agent({self.game_id})({moved_color_str}): WON! (Opponent king captured)")
            else:
                self.agent_won = False
                self.game_over_reason = "own_king_captured"
                print(f"agent({self.game_id})({moved_color_str}): LOSE! (Own king captured)")

        reward = self.get_reward(moved_color, moved_color_str)

        if not self.game_over:
            self.current_agent_color = not self.current_agent_color
            self.color_str = "WHITE" if self.current_agent_color == chess.WHITE else "BLACK"

        return reward, self.game_over, False

    def get_reward(self, moved_color, moved_color_str):
        reward = 0.0
        if not self.last_move_was_legal:
            print(f"agent({self.game_id})({moved_color_str}): ILLEGAL move!")
            return -10.0

        # Если ходил WHITE, добавляем repeat_penalty_white
        # Если ходил BLACK, добавляем repeat_penalty_black
        if moved_color == chess.WHITE:
            reward += getattr(self, 'repeat_penalty_white', 0.0)
        else:
            reward += getattr(self, 'repeat_penalty_black', 0.0)

        if self.moved_king:
            reward = -1.
        if self.did_castle:
            reward += 1.0

        if self.captured_piece is not None:
            # Определяем значение фигуры (например, используя словарь piece_values)
            val = piece_values.get(self.captured_piece.piece_type, 0)
            if self.captured_piece.color == moved_color:
                reward -= val  # потеря своей фигуры
            else:
                reward += val  # захват чужой фигуры

        if self.captured_piece and self.captured_piece.piece_type == chess.KING:
            if self.agent_won:
                reward += 10.0
            else:
                reward -= 10.0
            return reward

        if self.game_over:
            return reward

        # Дополнительные проверки: 1 фигура, 2 фигуры (2 короля) и т.д.
        pieces_on_board = list(self.board.piece_map().values())
        if len(pieces_on_board) == 1:
            self.game_over = True
            self.agent_won = False
            self.game_over_reason = "only_1_piece_left"
            reward -= 10.0
            print(f"agent({self.game_id})({moved_color_str}): LOSE! (Only 1 piece left on board)")
            return reward

        if len(pieces_on_board) == 2:
            self.game_over = True
            self.game_over_reason = "only_2_kings_draw"
            print(f"agent({self.game_id})({moved_color_str}): DRAW! (Only kings left)")
            return reward

        white_pieces = [p for p in pieces_on_board if p.color == chess.WHITE]
        black_pieces = [p for p in pieces_on_board if p.color == chess.BLACK]
        white_non_king = [p for p in white_pieces if p.piece_type != chess.KING]
        black_non_king = [p for p in black_pieces if p.piece_type != chess.KING]

        if len(white_non_king) == 0:
            self.game_over = True
            if moved_color == chess.WHITE:
                self.agent_won = False
                self.game_over_reason = "white_only_king_left"
                reward -= 10.0
                print(f"agent({self.game_id})(WHITE): LOSE! (White has only the king)")
            else:
                self.agent_won = True
                self.game_over_reason = "black_wins_white_stripped"
                reward += 10.0
                print(f"agent({self.game_id})(BLACK): WON! (White has only the king)")
            return reward

        if len(black_non_king) == 0:
            self.game_over = True
            if moved_color == chess.BLACK:
                self.agent_won = False
                self.game_over_reason = "black_only_king_left"
                reward -= 10.0
                print(f"agent({self.game_id})(BLACK): LOSE! (Black has only the king)")
            else:
                self.agent_won = True
                self.game_over_reason = "white_wins_black_stripped"
                reward += 10.0
                print(f"agent({self.game_id})(WHITE): WON! (Black has only the king)")
            return reward

        return reward

    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def process_moves(games, actions):
        rewards, dones, illegal_moves = [], [], []
        agent_wons, reasons, color_strs = [], [], []
        valid_moves_count, king_caps, king_missed_list = 0, [], []

        for g, act in zip(games, actions):
            r, d, ill = g.play_move(act)
            rewards.append(r); dones.append(d); illegal_moves.append(ill)
            agent_wons.append(g.agent_won)
            reasons.append(g.game_over_reason)
            color_strs.append(g.color_str)
            if g.last_move_was_legal: valid_moves_count += 1

            # record king or queen capture/loss
            is_king_cap  = d and g.game_over_reason=="opponent_king_captured"
            qp = g.captured_piece
            is_queen_cap = bool(qp and qp.piece_type==chess.QUEEN)
            king_caps.append(is_king_cap or is_queen_cap)

            # king_caps.append(d and g.game_over_reason == "opponent_king_captured")

            king_missed_list.append(g.king_missed); g.king_missed = False
            if d: g.reset()

        return (games, rewards, dones, agent_wons, reasons, color_strs,
                valid_moves_count, illegal_moves, king_caps, king_missed_list)

def states_board_and_masks(games, device='mps'):
    """
    Формируем батч состояний [batch, 65,16]
    и масок допустимых ходов [batch,4096].
    Также возвращаем список boards (game.board).
    """
    states_tensor = torch.stack([g.update() for g in games]).float().to(device)
    # print("states_tensor: ", states_tensor.shape)
    boards = [g.board for g in games]
    masks_list = []
    for board in boards:
        moves = list(board.generate_pseudo_legal_moves())
        mask = torch.zeros(4096, dtype=torch.bool, device=device)
        for mv in moves:
            idx = move_to_index(mv)
            mask[idx] = True
        masks_list.append(mask)
    masks_tensor = torch.stack(masks_list)
    return states_tensor, boards, masks_tensor










