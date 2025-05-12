import pygame
import chess
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import os

from Model import ChessModel, index_to_move
from Chess_background import ChessGame, states_board_and_masks

pygame.init()

WIDTH, HEIGHT = 800, 800
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Шахматы')
SQ_SIZE = WIDTH // 8

def load_images():
    pieces = ['wp', 'bp', 'wr', 'br', 'wn', 'bn', 'wb', 'bb', 'wq', 'bq', 'wk', 'bk']
    images = {}
    for piece in pieces:
        img = pygame.image.load(os.path.join("images", f"{piece}.png"))
        images[piece] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
    return images

IMAGES = load_images()

def draw_board(screen, board):
    colors = [pygame.Color(255, 255, 255), pygame.Color(50, 50, 50)]
    for r in range(8):
        for c in range(8):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
            piece = board.piece_at(chess.square(c, 7 - r))
            if piece:
                key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().lower()
                screen.blit(IMAGES[key], pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_end_game_message(screen, game, agent_won, mv, reason, color_str):
    font = pygame.font.SysFont("Arial", 30, bold=True)
    overlay = pygame.Surface((WIDTH, 120), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, HEIGHT // 2 - 60))
    if agent_won:
        text1 = font.render(f"{color_str} (Agent) Won!", True, pygame.Color(255, 255, 255))
    else:
        text1 = font.render(f"{color_str} (Agent) Lost!", True, pygame.Color(255, 0, 0))
    text2 = font.render(f"Reason: {reason} by move {mv}", True, pygame.Color(255, 200, 200))
    screen.blit(text1, text1.get_rect(center=(WIDTH//2, HEIGHT//2 - 20)))
    screen.blit(text2, text2.get_rect(center=(WIDTH//2, HEIGHT//2 + 20)))

def reset_game():
    return ChessGame(game_id=1)

def main():
    device = torch.device('cpu')

    model = ChessModel().to(device)
    ckpt = 'chess_model_transformer_weights_exp2.pth'   # ← обновлённый путь
    if os.path.exists(ckpt):
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading model:", e)
    else:
        print("No model checkpoint found.")
    model.eval()

    game = ChessGame(game_id=1)
    clock = pygame.time.Clock()
    running = True
    agent_won = None
    mv = None
    reason = None
    color_str = ''

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game.game_over:
            draw_end_game_message(WINDOW, game, agent_won, mv, reason, color_str)
            pygame.display.flip()
            waiting = True
            while waiting and running:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_r:
                            game = reset_game()
                            waiting = False
                        elif ev.key == pygame.K_d:
                            running = False
                            waiting = False
                clock.tick(1)
            continue

        draw_board(WINDOW, game.board)
        pygame.display.flip()
        clock.tick(2)

        # 1) get state & mask
        states_tensor, _, masks_tensor = states_board_and_masks([game], device)

        # 2) get and move hidden token to CPU
        token_in = game.get_current_token().to(device).unsqueeze(0)  # shape [1,512]

        # 3) forward in no_grad
        with torch.no_grad():
            logits, token_next, _ = model(states_tensor, token_in, masks_tensor)

        # 4) save token back
        game.set_current_token(token_next.squeeze(0))

        # 5) sample action
        probs = F.softmax(logits, dim=1)
        dist = Categorical(probs)
        action = dist.sample().item()
        mv = index_to_move(action)

        # 6) step environment
        (games, rewards, dones, agent_wons,
         reasons, color_strs, valid_count, _,
         king_caps, _) = ChessGame.process_moves([game], [action])

        game       = games[0]
        agent_won  = agent_wons[0]
        reason     = reasons[0]
        color_str  = color_strs[0]

        if dones[0]:
            game.game_over = True
            continue

        print(f"Move: {mv} Color: {color_str} Reward: {rewards[0]}")

    pygame.quit()

if __name__ == '__main__':
    main()
