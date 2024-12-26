import numpy as np

import gym
from gym import spaces

from copy import deepcopy

# Define the Tic-Tac-Toe Environment
class TicTacToeEnv(gym.Env):
    def __init__(self, max_turns=10):

        self.observation_space = spaces.Box(low=-1, high=1, shape=(9, ), dtype=int)

        self.action_space = spaces.Discrete(9)

        self.max_turns = max_turns

        # self.opponent_action_space = spaces.Discrete(9)

    def _get_obs(self):
        return {
            'board': deepcopy(self.board),
        }
    
    def _get_info(self):
        return {
            'winner': self._winner()
        }

    def _winner(self):
        # if len(self.valid_actions()) < 2:
        #     winner = None
        if self.is_winning(self.board, 1): winner = 1
        elif self.is_winning(self.board, -1): winner = -1
        else:
            winner = None

        return winner
    
    def reset(self, seed=42):
        self.board = np.zeros(9)  # Flattened 3x3 board
        self.turns = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, player_action, opponent_action):

        # Action and reward
        if player_action in self.valid_actions() and opponent_action in self.valid_actions():
            # and opponent_action in self.valid_actions():
                
            if player_action != opponent_action:
                self.board[player_action] = 1
                self.board[opponent_action] = -1
                reward = -0.1      # Reward for regular moves
            else:
                reward = 0     # Avoid playing on the same cell as opponent
        
        elif opponent_action in self.valid_actions():
            reward = -5         # for invalid action -5 for train

        elif player_action in self.valid_actions():
            reward = -0.1          # reward for regular moves

        else:
            reward = -5         # if both are invalid focus on player making invalid move.


        # Next state and reward
        observation = self._get_obs()
        info = self._get_info()
        winner = info['winner']
        
        if len(self.valid_actions()) < 2 \
        or winner:
            terminated = True

            if self.is_winning(self.board, -1) and self.is_winning(self.board, 1):
                reward = 0 # for draw
                info['winner'] = None
            elif winner == 1: reward = 100 # for winning 10 for train
            elif winner == -1: reward = -20 # for losing -10 FOR TRAIN

        else: terminated = False

        self.turns += 1

        if self.turns >= self.max_turns:
            terminated = True
            reward = -10
            winner = None

        # self.render()

        # The extra False is for 'truncated'
        return observation, reward, terminated, False, info 

    def valid_actions(self):
        return [i for i, x in enumerate(self.board) if x == 0]
    
    def is_winning(self, board, player):
        # Check all win conditions
        wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for win in wins:
            if all(board[i] == player for i in win):
                return True
        return False

    def render(self):
        return self.board.reshape([3,3])