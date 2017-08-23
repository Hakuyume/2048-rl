import argparse
import numpy as np
import os
import random

import chainer
import chainerrl

from g2048 import G2048
from net import CNN


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
    parser.add_argument('--init')
    parser.add_argument('--resume')
    parser.add_argument('--out', default='agent')
    args = parser.parse_args()

    game = G2048()

    def random_action():
        return random.choice(np.nonzero(game.movability)[0])

    model = Agent(CNN())
    if args.init:
        chainer.serializers.load_npz(args.init, model.model)
    else:
        model.model(np.zeros((1, 4, 4)))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    if args.init:
        optimizer = chainer.optimizers.MomentumSGD(lr=0.01)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
    else:
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

    episode = 0

    if args.resume:
        agent.load(args.resume)
        replay_buffer.load(os.path.join(args.resume, 'replay_buffer.pkl'))
        misc = np.load(os.path.join(args.resume, 'misc.npz'))
        episode = misc['episode']

    while True:
        episode += 1

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
                episode, game.score,
                1 << game.board.max(), agent.get_statistics()))
        agent.stop_episode_and_train(game.board.copy(), -game.score, True)

        if episode % 1000 == 0:
            agent.save(args.out)
            replay_buffer.save(os.path.join(args.out, 'replay_buffer.pkl'))
            np.savez(os.path.join(args.out, 'misc.npz'), episode=episode)

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
