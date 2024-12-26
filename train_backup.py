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
from model import minimaxDQN, DQN
from dataset import ReplayDataset

# Create Environment
BUFFER_SIZE = 100
EPISODES = 200
BATCH_SIZE = 32
# EPOCHS = 200

LR = 1e-2
GAMMA = 0.2 # Decrease gamme to make the model settle in quicker
LR_STEP_SIZE = 10 # Increase step size if model settles too quick

EPS = 1.0
EPS_MIN = 0.3
EPS_DECAY = 0.95 # Decrease if UPDATE_EVERY is low

UPDATE_EVERY = 5 # keep low if LR high

env = TicTacToeEnv()
test_env = TicTacToeEnv()

model = DQN()

target_model = DQN()
target_model.load_state_dict(model.state_dict())
target_model.eval()

# Collect data in replay buffer


buffer = ReplayDataset(BUFFER_SIZE)

opponent = Opponent()
test_opp = Opponent()

# opponent = RuleBasedOpponent()
# test_opp = RuleBasedOpponent()




optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=GAMMA)

plt.ion()
fig, axs = plt.subplots(1,4,figsize=(15,5))

model.train()
train_history = []

# f = open("env.txt", "w")
for episode in range(EPISODES):
    terminated = False
    observation, info = env.reset()

    train_loss = 0

    while not terminated:
        state = observation['board']
        if np.random.random() <= EPS:
            player_action = env.action_space.sample()
        else:
            player_action = model.choose_action(state)

        opponent_action = opponent.choose_action(env)
        # opponent_action = random.sample(env.valid_actions(), 1)[0]

        
        observation, reward, terminated, _, info  = env.step(player_action, opponent_action)
        next_state = observation['board']

        # f.write(f"{player_action}, {opponent_action}\n")
        # for row in env.render():
        #     f.write(f"{row.tolist()}\n")
        # f.write("#"*140)
        # f.write("\n")

        buffer.push((state, player_action, opponent_action, reward, next_state, terminated))

# f.close()

        dataloader = DataLoader(buffer, batch_size=BATCH_SIZE, shuffle=True)

    # for epoch in range(EPOCHS):
        if len(buffer) >= BATCH_SIZE:
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()

            # for i in range(5): # Try to make multiple gradient steps for each batch
                pred, loss = model.predict(X, y, target_model)

                # print(loss.item())

                train_loss += loss.item()

                loss.backward()
                optimizer.step()

                break

            # scheduler.step()

    torch.save(model.state_dict(), 'saved_models/dqn2.pth')    

    train_history.append(train_loss)

    axs[0].cla()
    axs[0].plot(train_history)

    axs[0].set_xlim(-3, EPISODES)
    axs[0].set_ylim(-3, max(train_history)+10)

    if episode % UPDATE_EVERY == 0:
            target_model.load_state_dict(model.state_dict())

            EPS = max(EPS_MIN, EPS * EPS_DECAY)
            print(EPS)


    # state1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # state2 = [-1, 0, 1, 0, -1, 0, 0, 0, 1]

    observation, info = test_env.reset()

    if isinstance(model, DQN):
        state = observation['board']
        
        test_opp_action = test_opp.choose_action(test_env)
        model.plot_q_vals(axs[1], state, test_opp_action)

        # print(state, model.choose_action(state), test_opp_action, reward)

        observation, reward, terminated, _, info = test_env.step(model.choose_action(state, True), test_opp_action)


        state = observation['board']
        test_opp_action = test_opp.choose_action(test_env)
        model.plot_q_vals(axs[2], state, test_opp_action)

        # print(state, model.choose_action(state), test_opp_action, reward)

        observation, reward, terminated, _, info = test_env.step(model.choose_action(state, True), test_opp_action)

        state = observation['board']
        test_opp_action = test_opp.choose_action(test_env)
        model.plot_q_vals(axs[3], state, test_opp_action)

        # print(state, model.choose_action(state), test_opp.choose_action(test_env), reward)

        observation, reward, terminated, _, info = test_env.step(model.choose_action(state, True), test_opp_action)

        # print(state, model.choose_action(state), test_opp_action)

    # elif isinstance(model, minimaxDQN):
    #     model.plot_q_vals(axs[1], state1, 0)    
    #     model.plot_q_vals(axs[2], state2, 1)

    fig.canvas.draw()
    fig.canvas.flush_events()

input()