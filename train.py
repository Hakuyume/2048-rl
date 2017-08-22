import argparse
import numpy as np
import os
import random

import chainer
import chainerrl

from g2048 import G2048
from net import Net


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
    parser.add_argument('--episodes', type=int, default=10000)
    args = parser.parse_args()

    game = G2048()

    def random_action():
        return random.choice(np.nonzero(game.movability)[0])

    model = Agent(Net())
    model(np.zeros((1, 4, 4)))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(model)

    gamma = 0.95
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.1, random_action_func=random_action)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1 << 20)
    agent = chainerrl.agents.DoubleDQN(
        model, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1,
        target_update_interval=100)

    if os.path.exists('agent') and os.path.isdir('agent'):
        agent.load('agent')

    for i in range(args.episodes):
        game.reset()
        reward = 0
        while not game.is_finished:
            action = agent.act_and_train(game.board.copy(), reward)
            if game.movability[action]:
                prev = game.score
                game.move(action)
                reward = game.score - prev
            else:
                reward = -1
        print(
            '{:d}: score: {:d}, max: {:d}, stat: {}'
            .format(
                i, game.score,
                1 << game.board.max(), agent.get_statistics()))
        agent.stop_episode_and_train(game.board.copy(), -game.score, True)

        if i % 1000 == 0:
            agent.save('agent')

            for _ in range(10):
                game.reset()
                while not game.is_finished:
                    action = agent.act(game.board.copy())
                    if not game.movability[action]:
                        action = random_action()
                    game.move(action)
                print('Test: score: {:d}, max: {:d}'.format(
                    game.score, 1 << game.board.max()))
                agent.stop_episode()
