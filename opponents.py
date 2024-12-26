import numpy as np

class Opponent:
    def choose_action(self, env):
        board = env.board

        # Number of Os played.
        n_os = np.sum(board == -1).item()

        if n_os == 0:
            return 0
        elif n_os == 1:
            if board[2] != 1 and 1 in env.valid_actions(): return 1
            else: return 4
        elif n_os == 2:
            if board[1] == -1:
                if 2 in env.valid_actions(): return 2
                return np.random.choice(env.valid_actions()).item()
            elif board[4] == -1:
                if 8 in env.valid_actions(): return 8
                return np.random.choice(env.valid_actions()).item()
            else:
                print("This case wasn't accounted for :3")
                return np.random.choice(env.valid_actions()).item()
            
        else:
            # print("Random Moves until finish")
            return np.random.choice(env.valid_actions()).item()
        
class RuleBasedOpponent:
    def choose_action(self, env):
        board = env.board
        # Prioritize winning moves or blocking the player
        for action in range(9):
            if board[action] == 0:
                # Check if the move wins for the opponent
                temp_board = board.copy()
                temp_board[action] = -1
                if env.is_winning(temp_board, -1):
                    return action
                # Check if the move blocks the player
                temp_board[action] = 1
                if env.is_winning(temp_board, 1):
                    return action
        # Otherwise, pick a random valid move
        return np.random.choice(env.valid_actions()).item()