import numpy as np

class Connect4Board:
    ROWS = 6
    COLS = 7
    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
    def make_move(self, col, player):
        for r in reversed(range(self.ROWS)):
            if self.board[r, col] == 0:
                self.board[r, col] = player
                return True
        return False
    def get_valid_moves(self):
        return [c for c in range(self.COLS) if self.board[0, c] == 0]
    def is_full(self):
        return np.all(self.board != 0)

