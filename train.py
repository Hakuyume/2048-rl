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

    game = G2048()

    def random_action():
        return random.choice(np.nonzero(game.is_movable)[0])

    model = MLP(args.units)
    model(np.zeros((1, 4, 4)))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(model)

    gamma = 0.95
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=random_action)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1 << 20)
    agent = chainerrl.agents.DoubleDQN(
        model, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1,
        target_update_interval=100)

    for i in range(args.episodes):
        game.reset()
        reward = 0
        while not game.is_finished:
            action = agent.act_and_train(game.board, reward)
            prev = game.board.max()
            game.move(action)
            reward = game.board.max() - prev
        print(
            '{:d}: score: {:d}, max: {:d}, stat: {}'
            .format(
                i, game.score,
                1 << game.board.max(), agent.get_statistics()))
        agent.stop_episode_and_train(game.board, reward, True)

    for i in range(10):
        game.reset()
        while not game.is_finished:
            if random.uniform(0, 1) > 1 / 100:
                action = agent.act(game.board)
            else:
                action = random_action()
            game.move(action)
        print('{:d}: score: {:d}, max: {:d}'.format(
            i, game.score, 1 << game.board.max()))
        agent.stop_episode()
