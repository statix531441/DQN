import random
from collections import deque

import numpy as np

import torch
from torch.utils.data import Dataset


class ReplayDataset(Dataset):
    def __init__(self, capacity=10000):
        self.buffer = deque()
        self.capacity = capacity

    def clear(self):
        self.buffer = []

    def push(self, replay):
        self.buffer.append(replay)

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        state, player_action, opponent_action, opponent_num, reward, next_state, terminated = self.buffer[idx]

        state = torch.tensor(state).to(torch.float)
        # player_action = int(player_action)
        # opponent_action = int(opponent_action)
        reward = torch.tensor(reward).to(torch.float)
        next_state = torch.tensor(next_state).to(torch.float)
        terminated = torch.tensor(terminated).to(torch.float)

        X = {
            'state' : state,
            'player_action' : player_action, 
            'opponent_action' : opponent_action, 
            'opponent_num' : opponent_num
        }
        y = {
            'reward' : reward, 
            'next_state' : next_state, 
            'terminated' : terminated
        }
        return X, y
    
    def sample(self, batch_size):
        batch =  random.sample(self.buffer, k=batch_size)

        state, player_action, opponent_action, opponent_num, reward, next_state, terminated = zip(*batch)

        state, player_action, opponent_action, opponent_num, reward, next_state, terminated = np.array(state), np.array(player_action), np.array(opponent_action), np.array(opponent_num), np.array(reward), np.array(next_state), np.array(terminated)

        state = torch.tensor(state).to(torch.float)
        # player_action = int(player_action)
        opponent_action = torch.tensor(opponent_action)
        opponent_num = torch.tensor(opponent_num)
        reward = torch.tensor(reward).to(torch.float)
        next_state = torch.tensor(next_state).to(torch.float)
        terminated = torch.tensor(terminated).to(torch.float)

        X = {
            'state' : state,
            'player_action' : player_action, 
            'opponent_action' : opponent_action,
            'opponent_num': opponent_num,
        }
        y = {
            'reward' : reward, 
            'next_state' : next_state, 
            'terminated' : terminated
        }
        return X, y
    
    def ordered_buffer(self):
        batch = self.buffer

        state, player_action, opponent_action, opponent_num, reward, next_state, terminated = zip(*batch)

        state, player_action, opponent_action, opponent_num, reward, next_state, terminated = np.array(state), np.array(player_action), np.array(opponent_action), np.array(opponent_num), np.array(reward), np.array(next_state), np.array(terminated)

        state = torch.tensor(state).to(torch.float)
        # player_action = int(player_action)
        opponent_action = torch.tensor(opponent_action)
        opponent_num = torch.tensor(opponent_num)
        reward = torch.tensor(reward).to(torch.float)
        next_state = torch.tensor(next_state).to(torch.float)
        terminated = torch.tensor(terminated).to(torch.float)

        X = {
            'state' : state,
            'player_action' : player_action, 
            'opponent_action' : opponent_action,
            'opponent_num': opponent_num,
        }
        y = {
            'reward' : reward, 
            'next_state' : next_state, 
            'terminated' : terminated
        }
        return X, y



    