import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


class MLP(chainer.Chain):

    def __init__(self, n_unit=256):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_unit)
            self.l2 = L.Linear(n_unit)
            self.l3 = L.Linear(4)

    def __call__(self, x):
        xp = self.xp

        h = (x[:, None] == (xp.arange(16) + 1)[:, None, None]) \
            .astype(np.float32)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        y = self.l3(h)
        return y
