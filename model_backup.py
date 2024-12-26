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
            

        # def init_weights_zero(module):
        #     if isinstance(module, nn.Linear):
        #         module.weight.fill_(0)
        #         module.bias.fill_(0)

        # with torch.no_grad():
        #     self.fc.apply(init_weights_zero)

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

    def choose_actions(self, state, opponent_action=None, best=False):
        
        # player_policy, opponent_policy = self.return_policies(state)

        # if best:
        #     return int(np.argmax(player_policy).item()), np.random.choice(np.arange(9), p=opponent_policy)
        # return np.random.choice(np.arange(9), p=player_policy), np.random.choice(np.arange(9), p=opponent_policy)

        q_values = self.forward(torch.tensor(state).to(torch.float)).reshape((9, 9)).detach()

        if opponent_action is not None:
            player_policy = F.softmax(q_values[:, opponent_action], dim=0).detach().numpy()

        else:
            player_policy = F.softmax(torch.min(q_values, axis=1).values, dim=0).detach().numpy().astype(np.float64)
            opponent_policy = F.softmax(torch.min(-q_values, axis=0).values, dim=0).detach().numpy().astype(np.float64)
            opponent_policy /= opponent_policy.sum()

            opponent_action = np.random.choice(np.arange(9), p=opponent_policy)
            # opponent_action = torch.min(torch.max(q_values, axis=1).values, axis=0)


        # Workaround for numpy bug https://github.com/numpy/numpy/pull/6131
        player_policy /= player_policy.sum()
            
        if best: player_action = np.argmax(player_policy).item()
        else: player_action = np.random.choice(np.arange(9), p=player_policy)

        

        return int(player_action), int(opponent_action)






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
        with torch.no_grad():
            q_vals = self.forward(torch.tensor(state).to(torch.float))
            best_action = torch.argmax(q_vals).item()
            ax.set_title(f"{best_action}, {opponent_action}")
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

            