import argparse
import numpy as np
import random

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

from g2048 import G2048


class MLP(chainer.Chain):

    def __init__(self, n_unit):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_unit)
            self.l2 = L.Linear(n_unit)
            self.l3 = L.Linear(4)

    def __call__(self, x):
        xp = self.xp
        h = (x[:, np.newaxis] == xp.arange(16)[:, np.newaxis, np.newaxis]) \
            .astype(np.float32)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--units', type=int, default=64)
    args = parser.parse_args()

    model = MLP(args.units)
    model(np.zeros((1, 4, 4)))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    gamma = 0.95
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3,
        random_action_func=lambda: random.randrange(4))
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1 << 20)
    agent = chainerrl.agents.DoubleDQN(
        model, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1,
        target_update_interval=100)

    for i in range(args.episodes):
        g = G2048()
        reward = 0
        while not g.is_finished:
            action = agent.act_and_train(g.board, reward)
            prev = g.score
            g.move(action)
            reward = g.score - prev
        print(
            '{:d}: score: {:d}, stat: {}'
            .format(i, g.score, agent.get_statistics()))
        agent.stop_episode_and_train(g.board, reward, True)

    for i in range(10):
        g = G2048()
        for t in range(200):
            action = agent.act(g.board)
            g.move(action)
            if g.is_finished:
                break
        print('{:d}: score: {:d}'.format(i, g.score))
        agent.stop_episode()
