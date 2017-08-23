import argparse
import numpy as np

import chainer
import chainer.functions as F
from chainer.training import extensions

from g2048 import G2048
from net import CNN


class MultiLabelClassifier(chainer.Chain):

    def __init__(self, model):
        super().__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, x, t):
        y = self.model(x)
        t = t.astype(np.int32)

        loss = F.sigmoid_cross_entropy(y, t)
        accuracy = ((y.data >= 0) == t).mean()
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss


class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __len__(self):
        return self.n_sample

    def get_example(self, i):
        game = G2048()
        game.reset()
        game.board[:] = np.random.randint(0, 16, size=game.board.shape)

        d = np.random.randint(0, 3)
        while game.movability[d]:
            game.move(d)
        return game.board, game.movability


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--out', default='result')
    args = parser.parse_args()

    model = MultiLabelClassifier(CNN())

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    dataset = Dataset(2048)
    iter_ = chainer.iterators.SerialIterator(dataset, args.batchsize)

    print(
        'chance rate: ',
        sum(dataset[i][1].mean() for i in range(len(dataset))) / len(dataset))

    updater = chainer.training.StandardUpdater(
        iter_, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(
        updater, (15000, 'iteration'), out=args.out)
    trainer.extend(
        extensions.snapshot_object(
            model.model, filename='model_iter_{.updater.iteration}'),
        trigger=(15000, 'iteration'))

    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=0.1),
        trigger=(10000, 'iteration'))

    log_interval = (10, 'iteration')
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(
        extensions.LogReport(trigger=log_interval))
    trainer.extend(
        extensions.PrintReport(
            ['iteration', 'lr', 'main/loss', 'main/accuracy']))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()
