import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer,DQN
from helper import plot
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.001
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 150
TARGET_UPDATE = 5

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = DQN(4)
        self.target_net = DQN(4)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()
        self.steps_done = 0
        self.trainer = QTrainer(self.model,self.target_net, lr=LR, gamma=self.gamma)
        self.stacked_state = deque(maxlen=4)


    def get_state(self, game:SnakeGameAI):
        game_board = game.get_map()

        while(len(self.stacked_state)<4):
            self.stacked_state.append(game_board)
        
        self.stacked_state.append(game_board)

        return torch.cat(list(self.stacked_state)).unsqueeze(0) #(B,C,W,H)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(torch.cat(states), actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample <= eps_threshold:
            move = random.randint(0, 3)
        else:
            prediction = self.model(state)
            move = torch.argmax(prediction).item()

        return move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        if(done == False):
            state_new = agent.get_state(game)
        else:
            state_new = None
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.stacked_state = deque(maxlen=4)
            agent.n_games += 1
            if(agent.n_games>=BATCH_SIZE):
                agent.train_long_memory()

            if agent.n_games % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.model.state_dict())

            if score > record:
                record = score
                #agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()