import numpy as np
import random


class G2048(object):

    def __init__(self, size=4):
        self.size = size
        self.score = 0
        self.board = np.zeros((size, size), dtype=np.uint8)

        for _ in range(2):
            self._add()

    @property
    def is_finished(self):
        for u in range(self.size):
            for v in range(self.size):
                if self.board[u, v] == 0:
                    return False
                if u > 0 and self.board[u - 1, v] == self.board[u, v]:
                    return False
                if v > 0 and self.board[u, v - 1] == self.board[u, v]:
                    return False
        return True

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
