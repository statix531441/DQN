import random
from sympy import Matrix

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tictac_backup import TicTacToeEnv
from opponents import Opponent, RuleBasedOpponent
from model import minimaxDQN
from dataset import ReplayDataset

from tqdm import tqdm

# Create Environment
BUFFER_SIZE = 500
EPISODES = 32768
BATCH_SIZE = 64
# EPOCHS = 200

LR = 1e-2
GAMMA =  0.1 # Decrease gamma to make the model settle in quicker
LR_STEP_SIZE =  4096 # Increase step size if model settles too quick

EPS = 1.0
EPS_MIN = 0.3
EPS_DECAY = 0.9 # Decrease if UPDATE_EVERY is low
EPS = EPS / EPS_DECAY

UPDATE_EVERY = 2048 # keep low if LR high

env = TicTacToeEnv()
test_env = TicTacToeEnv()

model = minimaxDQN()

target_model = minimaxDQN()
target_model.load_state_dict(model.state_dict())
target_model.eval()

# Collect data in replay buffer


buffer = ReplayDataset(BUFFER_SIZE)


opponent = RuleBasedOpponent()
test_opp = RuleBasedOpponent()
test_opp = Opponent()



optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=GAMMA)

plt.ion()
fig, axs = plt.subplots(1,4,figsize=(15,5))

model.train()
train_history = []

for episode in tqdm(range(EPISODES)):
    terminated = False
    observation, info = env.reset()

    train_loss = 0

    while not terminated:
        state = observation['board']
        if np.random.random() <= EPS:
            player_action = env.action_space.sample()
            # opponent_action = random.sample(env.valid_actions(), 1)[0]         # This basically makes it learn by playing against itself.
            # opponent_action = random.sample(env.valid_actions(), 1)[0]
            opponent_action = opponent.choose_action(env)                    # Consider this to learn both general and specific strats
        else:
            player_action, opponent_action = model.choose_actions(state)
            # opponent_action = env.action_space.sample()
            opponent_action = opponent.choose_action(env)
        
        observation, reward, terminated, _, info  = env.step(player_action, opponent_action)
        # print(player_action, opponent_action)

        next_state = observation['board']

        buffer.push((state, player_action, opponent_action, reward, next_state, terminated))


        # dataloader = DataLoader(buffer, batch_size=BATCH_SIZE, shuffle=True)

    # for epoch in range(EPOCHS):
        if len(buffer) >= BATCH_SIZE:
            # for batch_idx, (X, y) in enumerate(dataloader):
            
            X, y = buffer.sample(BATCH_SIZE)
            
            optimizer.zero_grad()

        # for i in range(5): # Try to make multiple gradient steps for each batch
            pred, loss = model.predict(X, y, target_model)

            # print(loss.item())

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

    # if len(buffer) >= BATCH_SIZE: scheduler.step()

    torch.save(model.state_dict(), 'saved_models/minmax_dqn3.pth')    

    train_history.append(train_loss)

    if episode % UPDATE_EVERY == 0:
            target_model.load_state_dict(model.state_dict())

            EPS = max(EPS_MIN, EPS * EPS_DECAY)


    


# Monitoring
    axs[0].cla()
    axs[0].plot(train_history)

    axs[0].set_title(f"epsilon = {EPS}")

    axs[0].set_xlim(-3, EPISODES)
    axs[0].set_ylim(-3, max(train_history)+10)
    observation, info = test_env.reset()
    state = observation['board']

    # test_opp_action = test_opp.choose_action(test_env)
    # model.plot_q_vals(axs[1], state, test_opp_action)

    # # print(state, model.choose_actions(state, best=True)[0], test_opp_action, reward)

    # observation, reward, terminated, _, info = test_env.step(model.choose_actions(state, best=True)[0], test_opp_action)


    # state = observation['board']
    # test_opp_action = test_opp.choose_action(test_env)
    # model.plot_q_vals(axs[2], state, test_opp_action)

    # # print(state, model.choose_action(state), test_opp_action, reward)

    # observation, reward, terminated, _, info = test_env.step(model.choose_actions(state, best=True)[0], test_opp_action)

    # state = observation['board']
    # test_opp_action = test_opp.choose_action(test_env)
    # model.plot_q_vals(axs[3], state, test_opp_action)

    # # print(state, model.choose_action(state), test_opp.choose_action(test_env), reward)

    # observation, reward, terminated, _, info = test_env.step(model.choose_actions(state, best=True)[0], test_opp_action)

    # # print(state, model.choose_action(state), test_opp_action)

    fig.canvas.draw()
    fig.canvas.flush_events()

input()