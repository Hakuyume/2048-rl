import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


class CNN(chainer.Chain):

    def __init__(self, n_channel=256):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(n_channel, 3, pad=1)
            self.conv2 = L.Convolution2D(4, 3, pad=1)

    def __call__(self, x):
        xp = self.xp

        h = (x[:, None] == (xp.arange(1 + 16))[:, None, None]) \
            .astype(np.float32)
        h = F.relu(self.conv1(h))
        h = self.conv2(h)
        h = _global_max_pooling_2d(h)
        return F.reshape(h, (-1, 4))


def _global_max_pooling_2d(x):
    h = F.max(x, axis=2, keepdims=True)
    h = F.max(h, axis=3, keepdims=True)
    return h
