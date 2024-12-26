import random

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tictac import TicTacToeEnv
from opponents import Opponent, RuleBasedOpponent
from model import PolicyNet, minimaxDQN
from dataset import ReplayDataset

from tqdm import tqdm

# Create Environment
BUFFER_SIZE = 500
EPISODES = 1000
BATCH_SIZE = 64
# EPOCHS = 200

LR = 1e-3
# GAMMA =  0.1 # Decrease gamma to make the model settle in quicker
# LR_STEP_SIZE =  4096 # Increase step size if model settles too quick

EPS = 1.0
EPS_MIN = 0.3
EPS_DECAY = 0.95 # Decrease if UPDATE_EVERY is low
EPS = EPS / EPS_DECAY

UPDATE_EVERY = 20 # keep low if LR high

env = TicTacToeEnv()
test_env = TicTacToeEnv()

player = minimaxDQN()
player.load_state_dict(torch.load("saved_models/minmax_dqn3.pth", weights_only=True))

model = PolicyNet()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=GAMMA)

opponents = [Opponent(), RuleBasedOpponent()]


# Monitoring stuff #
train_history = []
plt.ion()
fig, axs = plt.subplots(1,4,figsize=(15,5))
####################

buffer = ReplayDataset(BUFFER_SIZE)

for episode in tqdm(range(EPISODES)):

    terminated = False
    observation, info = env.reset()
    train_loss = 0

    opponent_num = random.choice(range(len(opponents)))

    while not terminated:
        # Player POV
        state = observation['board']
        if np.random.random() <= EPS:
            player_action = env.action_space.sample()
        else:
            player_action = player.choose_action(state)
        opponent_action = opponents[opponent_num].choose_action(env)
        
        observation, rewards, terminated, _, info  = env.step(player_action, opponent_action)
        next_state = observation['board']

        # Substitute player_action with opp_num
        buffer.push((state, player_action, opponent_action, opponent_num, rewards[0], next_state, terminated))

        # Opponent POV
        # buffer.push((-state, opponent_action, player_action, rewards[1], -next_state, terminated))

        if len(buffer) >= BATCH_SIZE:

            optimizer.zero_grad()

            X, y = buffer.sample(BATCH_SIZE)
            pred, loss = model.predict(X['state'], X['opponent_action'], X['opponent_num'])
            train_loss += loss.item()
            loss.backward()

            optimizer.step()

    # if len(buffer) >= BATCH_SIZE: scheduler.step()

    torch.save(model.state_dict(), 'saved_models/policynet.pth')    

    if episode % UPDATE_EVERY == 0:

        test_opp_num = random.choice(range(len(opponents)))
        test_opp = opponents[test_opp_num]

        EPS = max(EPS_MIN, EPS * EPS_DECAY)

        observation, info = test_env.reset()

        ##############################################################################################################################
        state = observation['board']

        test_opp_action = test_opp.choose_action(test_env)
        model.plot_policy(axs[1], state, test_opp_num, test_opp_action)

        observation, rewards, terminated, _, info = test_env.step(player.choose_action(state, best=False), test_opp_action)
        ##############################################################################################################################
        state = observation['board']

        test_opp_action = test_opp.choose_action(test_env)
        model.plot_policy(axs[2], state, test_opp_num, test_opp_action)

        observation, rewards, terminated, _, info = test_env.step(player.choose_action(state, best=True), test_opp_action)
        ##############################################################################################################################
        state = observation['board']

        test_opp_action = test_opp.choose_action(test_env)
        model.plot_policy(axs[3], state, test_opp_num, test_opp_action)

        observation, rewards, terminated, _, info = test_env.step(player.choose_action(state, best=True), test_opp_action)
        ##############################################################################################################################



# Monitoring
    train_history.append(train_loss)
    axs[0].cla()
    axs[0].plot(train_history)

    axs[0].set_title(f"epsilon = {EPS}")

    axs[0].set_xlim(-int(0.01*EPISODES), EPISODES)
    axs[0].set_ylim(-3, max(train_history)+10)


    # observation, info = test_env.reset()
    # state = observation['board']

    fig.canvas.draw()
    fig.canvas.flush_events()

input()