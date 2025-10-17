import tkinter as tk
from tkinter import messagebox

ROWS, COLS = 6, 7
player = 1
board = [[0 for _ in range(COLS)] for _ in range(ROWS)]

def drop_piece(col):
    global player
    for row in reversed(range(ROWS)):
        if board[row][col] == 0:
            board[row][col] = player
            color = "red" if player == 1 else "yellow"
            canvas.create_oval(col*80+5, row*80+5, col*80+75, row*80+75, fill=color)
            if check_win(player):
                messagebox.showinfo("Game Over", f"Player {player} wins!")
                reset_board()
            player = 2 if player == 1 else 1
            return
    messagebox.showwarning("Invalid", "That column is full!")

def check_win(p):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c+i] == p for i in range(4)): return True
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r+i][c] == p for i in range(4)): return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r+i][c+i] == p for i in range(4)): return True
            if all(board[r+3-i][c+i] == p for i in range(4)): return True
    return False

def reset_board():
    global board, player
    board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    player = 1
    canvas.delete("all")
    draw_grid()

def draw_grid():
    for c in range(COLS):
        for r in range(ROWS):
            canvas.create_rectangle(c*80, r*80, c*80+80, r*80+80, outline="blue")

root = tk.Tk()
root.title("Connect 4")

canvas = tk.Canvas(root, width=COLS*80, height=ROWS*80, bg="white")
canvas.pack()

draw_grid()

frame = tk.Frame(root)
frame.pack()

for i in range(COLS):
    btn = tk.Button(frame, text=f"Drop {i+1}", command=lambda c=i: drop_piece(c))
    btn.pack(side="left")

root.mainloop()
