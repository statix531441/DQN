
from tictac import TicTacToeEnv
from dataset import ReplayDataset

from copy import deepcopy

import numpy as np
import nashpy as nash

import torch
import torch.nn as nn

import torch.nn.functional as F

# minimax DQN Model
class minimaxDQN(nn.Module):
    def __init__(self, model_cfg={}):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.Lineaer(64, 64),
            # nn.ReLU(),
            nn.Linear(128, 81),
        )
        self.lossfn = nn.MSELoss()
    
    def forward(self, state):
        return self.fc(state)

    def predict(self, X, y, target_model=None):
        out = self.forward(X['state']).reshape((-1, 9, 9))

        if target_model is None:
            target_model = self

        with torch.no_grad():
            next_out = target_model.forward(y['next_state']).reshape((-1, 9, 9)).detach()


        q_values = out[torch.arange(out.shape[0]), X['player_action'], X['opponent_action']]
        target_q_values = y['reward'] \
                        + torch.max(torch.min(next_out, axis=2).values, axis=1).values * (1-y['terminated'])
        
        # for terminated in y['terminated']:
        #     if terminated:
        #         print(target_q_values)

        # q_values = out.reshape((-1, 9, 9))
        # target_q_values = q_values.detach().clone()
        # target_q_values[torch.arange(target_q_values.shape[0]), X['player_action'], X['opponent_action']] = y['reward'] \
        #                 + torch.max(torch.min(next_out, axis=2).values, axis=1).values * (1-y['terminated'])


        loss = self.lossfn(q_values, target_q_values)

        return q_values, loss
    
    def plot_q_vals(self, ax, state, opponent_action):
        with torch.no_grad():
            
            q_vals = self.forward(torch.tensor(state).to(torch.float))
            q_vals = q_vals.reshape((9, 9))[:, opponent_action].reshape((3,3)).detach()
            player_policy = F.softmax(q_vals.reshape(9,), dim=0).numpy()
            player_action = np.argmax(player_policy).item()

            ax.set_title(f"{player_action}, {opponent_action}")
            q_vals -= q_vals.min()
            q_vals /= q_vals.max()
            ax.imshow(q_vals)


    def return_reduced_qvalues(self, state, opponent_num=0, policy_net=None, depth=0):

        with torch.no_grad():
            security_qvals = self.forward(torch.tensor(state).to(torch.float)).detach().reshape((9, 9))
    
        if policy_net is None or depth==-1:
            return torch.min(security_qvals, axis=1).values
        
        else:
            if depth==0:
                opponent_policy = policy_net.return_policy(state, opponent_num)
                reduced_qvals = opponent_policy * security_qvals
                reduced_qvals = reduced_qvals.sum(axis=1)
                return reduced_qvals
            else:
                env = TicTacToeEnv()

                deep_q_values = torch.zeros_like(security_qvals)

                for i in range(9):
                    for j in range(9):
                        env.reset()
                        env.board = deepcopy(state)
                        observation, rewards, terminated, _, info  = env.step(i, j)
                        next_state = observation['board']

                        if terminated:
                            deep_q_values[i, j] = rewards[0]
                        else:
                            deep_q_values[i, j] = rewards[0] + torch.max(self.return_reduced_qvalues(next_state, opponent_num, policy_net, depth=depth-1))

                opponent_policy = policy_net.return_policy(state, opponent_num)
                reduced_qvals = opponent_policy * deep_q_values
                reduced_qvals = reduced_qvals.sum(axis=1)
                # print(reduced_qvals)
                return reduced_qvals


    def return_policy(self, state, opponent_num=0, policy_net=None, depth=0, opponent_action=None):

        
        if opponent_action is None:
            reduced_qvals = self.return_reduced_qvalues(state, opponent_num, policy_net, depth)
            policy = F.softmax(reduced_qvals, dim=0).detach().numpy().astype(np.float64)
        else:
            security_qvals = self.forward(torch.tensor(state).to(torch.float)).detach().reshape((9, 9))
            policy = F.softmax(security_qvals[:, opponent_action], dim=0).detach().numpy().astype(np.float64)

        # Workaround for numpy bug https://github.com/numpy/numpy/pull/6131
        policy /= policy.sum()
        return policy


    def choose_action(self, state, opponent_num=0, policy_net=None, depth=0, opponent_action=None, best=False):
        
        player_policy = self.return_policy(state, opponent_num, policy_net, depth, opponent_action)
            
        if best: player_action = np.argmax(player_policy).item()
        else: player_action = np.random.choice(np.arange(9), p=player_policy)

        return int(player_action)






class DQN(nn.Module):
    def __init__(self, model_cfg={}):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 9),
            # nn.LeakyReLU()
        )

        # def init_weights_zero(module):
        #     if isinstance(module, nn.Linear):
        #         module.weight.fill_(0)
        #         module.bias.fill_(0)

        # with torch.no_grad():
        #     self.fc.apply(init_weights_zero)

        self.lossfn = nn.MSELoss()
    
    def forward(self, state):
        out = self.fc(state)
        return out

    def predict(self, X, y, target_model=None):
        out = self.forward(X['state'])


        if target_model is None:
            target_model = self

        with torch.no_grad():
            next_out = target_model.forward(y['next_state']).detach()


        q_values = out[torch.arange(out.shape[0]), X['player_action']]
        target_q_values = y['reward'] \
                        + torch.max(next_out, axis=1).values * (1-y['terminated'])
        
        # q_values = out #.reshape((-1, 9))
        # target_q_values = q_values.detach().clone()
        # target_q_values[torch.arange(target_q_values.shape[0]), X['player_action']] = y['reward'] \
        #                 + torch.max(next_out, axis=1).values * (1-y['terminated'])
        
        # print(np.round(target_q_values, 2))

        loss = self.lossfn(q_values, target_q_values)

        return q_values, loss
    
    def plot_q_vals(self, ax, state, opponent_action):
        ax.axis('off')
        with torch.no_grad():
            q_vals = self.forward(torch.tensor(state).to(torch.float))
            best_action = torch.argmax(q_vals).item()
            ax.set_title(f"Best Move: {best_action}")
            q_vals = q_vals.reshape((3,3)).detach().numpy()
            q_vals -= q_vals.min()
            q_vals /= q_vals.max()
            ax.imshow(q_vals)

    def choose_action(self, state, best=False):
        with torch.no_grad():
            q_vals = self.forward(torch.tensor(state).to(torch.float))

            player_policy = F.softmax(q_vals, dim=0).detach().numpy()
            
            if best: return int(torch.argmax(q_vals).item()) # change q_vals to player_policy for consistency
            
            return np.random.choice(np.arange(9), p=player_policy)


class PolicyNet(nn.Module):
    def __init__(self, max_opponents=10, opponent_vec_dim = 9):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(9+opponent_vec_dim, 32),               # (9-dim state, 9-dim opponent_vec)
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 9),
            nn.LogSoftmax(dim=1)
        )

        self.opponent_vec = nn.Embedding(max_opponents, opponent_vec_dim)

        self.lossfn = nn.NLLLoss()


    def forward(self, state, opponent_num):
        opponent_vec = self.opponent_vec(opponent_num)
        inp = torch.concat((state, opponent_vec), dim=1)
        return self.policy(inp)
    

    def predict(self, state, opponent_action, opponent_num):

        log_policy = self.forward(state, opponent_num)

        loss = self.lossfn(log_policy, opponent_action)

        return torch.exp(log_policy), loss
    
    def return_policy(self, state, opponent_num=0):
        state = torch.tensor(state).view(-1, state.shape[0]).float()
        opponent_num = torch.tensor(opponent_num).view(1)

        log_policy = self.forward(state, opponent_num)[0]

        return torch.exp(log_policy).detach()

    def plot_policy(self, ax, state, opponent_num, opponent_action):
        policy = self.return_policy(state, opponent_num).detach().numpy().reshape((3,3))
        ax.imshow(policy)

        ax.set_title(f"opponent={opponent_num}, actual_action={opponent_action}")






    





            