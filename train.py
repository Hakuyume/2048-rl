import argparse
import numpy as np
import random

import chainer
import chainerrl

from g2048 import G2048
from net import MLP


class Agent(chainer.Chain):

    def __init__(self, model):
        super().__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, x):
        h = self.model(x)
        return chainerrl.action_value.DiscreteActionValue(h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--units', type=int, default=64)
    args = parser.parse_args()

    game = G2048()

    def random_action():
        return random.choice(np.nonzero(game.movability)[0])

    model = Agent(MLP())
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
