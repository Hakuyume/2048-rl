import numpy as np
import random


class G2048(object):

    def __init__(self, size=4):
        self.size = size
        self.board = np.empty((size, size), dtype=np.uint8)

    def reset(self):
        self.score = 0
        self.board[:] = 0
        for _ in range(2):
            self._add()

    @property
    def movability(self):
        m = np.zeros(4, dtype=bool)
        for d in range(4):
            board = np.rot90(self.board, d)
            if np.logical_and(board[:, :-1] == 0, board[:, 1:] > 0).any():
                m[d] = True
            elif np.logical_and(
                    board[:, :-1] > 0, board[:, :-1] == board[:, 1:]).any():
                m[d] = True
        return m

    @property
    def is_finished(self):
        return not self.movability.any()

    def _add(self):
        blank = tuple(zip(*np.where(self.board == 0)))
        if len(blank) > 0:
            u, v = random.choice(blank)
            if random.uniform(0, 1) > 1 / 4:
                self.board[u, v] = 1
            else:
                self.board[u, v] = 2

    def move(self, direction):
        change = False

        for line in np.rot90(self.board, direction):
            v, w = 0, 0
            new_line = np.zeros_like(line)
            while v < self.size:
                if line[v] == 0:
                    v += 1
                elif new_line[w] == line[v]:
                    new_line[w] += 1
                    self.score += 1 << new_line[w]
                    change = True
                    v += 1
                    w += 1
                elif new_line[w] == 0:
                    new_line[w] = line[v]
                    change = change or not v == w
                    v += 1
                else:
                    w += 1
            line[:] = new_line

        if change:
            self._add()

    def normalize(self):
        self.board[:] = min(
            (np.rot90(b, r)
             for b in (self.board, self.board.transpose())
             for r in range(4)),
            key=lambda b: tuple(b.flatten()))
