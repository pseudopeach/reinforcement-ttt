import numpy as np
import random


class TTT:
    def __init__(self):
        self.board = np.zeros((3, 3))

    def reset(self):
        self.board = np.zeros((3, 3))
        return self.board

    def open_spaces(self):
        return np.argwhere(self.board == 0)

    def is_legal(self, action):
        space = (action // 3, action % 3)
        return self.board_get(space) == 0

    def step(self, action):
        space = (action // 3, action % 3)

        self.mark_space(space)


        return (self.board, 1.0 if self.is_won() else 0.0, self.is_finished())

    def mark_space(self, space):
        mark = self.whose_turn()
        letter = 'X' if mark == 1 else 'O'
        # print('{} moves to {}'.format(letter, space))

        if self.board_get(space) != 0:
            raise Exception('Illegal action!')

        self.board[space[0]][space[1]] = mark

    def board_get(self, space):
        return self.board[space[0]][space[1]]

    def whose_turn(self):
        if self.open_spaces().shape[0] == 0:
            return None

        if self.open_spaces().shape[0] % 2 == 0:
            return -1
        else:
            return 1

    def is_won(self):
        for w in ALL_WINS:
            m_count = 0
            for space in w:
                m_count += self.board_get(space)

            if m_count == 3 or m_count == -3:
                return True

        return False

    def is_finished(self):
        return self.is_won() or self.open_spaces().shape[0] == 0



class TTT2:
    def __init__(self):
        self.env = TTT()
        self.ai_player_i = 1 if random.random() < 0.5 else -1
        # self.ai_player_i = -1
        if self.ai_player_i == 1:
            self.ai_move()

    def reset(self):
        self.env.reset()
        self.ai_player_i = 1 if random.random() < 0.5 else -1
        if self.ai_player_i == 1:
            self.ai_move()

        return self.env.board

    def step(self, action):
        if not self.env.is_legal(action):
            return -np.ones((3, 3)), -1, True

        new_state, reward, is_terminal = self.env.step(action)
        if is_terminal:
            return new_state, reward, is_terminal

        new_state, reward, is_terminal = self.ai_move()

        return new_state, -reward, is_terminal


    def ai_move(self):
        for w in ALL_WINS:
            opp_count = 0
            blank_count = 0
            blank = None
            for space in w:
                if self.env.board_get(space) == -1*self.ai_player_i:
                    opp_count += 1
                if self.env.board_get(space) == 0:
                    blank_count += 1
                    blank = space

            if opp_count == 0 and blank_count == 1:
                return self.env.step(action_for_space(blank))
            if opp_count == 2 and blank_count == 1:
                return self.env.step(action_for_space(blank))

        space = self.first_open([(1, 1), (0, 0), (0, 2), (2, 0), (2, 2)])
        if space:
            return self.env.step(action_for_space(space))

        return self.env.step(action_for_space(self.env.open_spaces()[0]))

    def first_open(self, spaces):
        for space in spaces:
            if self.env.board_get(space) == 0:
                return space

        return None

    def get_action_space_size(self):
        return 9


ALL_WINS = [
    [[0, 0], [0, 1], [0, 2]],
    [[1, 0], [1, 1], [1, 2]],
    [[2, 0], [2, 1], [2, 2]],

    [[0, 0], [1, 0], [2, 0]],
    [[0, 1], [1, 1], [2, 1]],
    [[0, 2], [1, 2], [2, 2]],

    [[0, 0], [1, 1], [2, 2]],
    [[0, 2], [1, 1], [2, 0]],
]


def action_for_space(space):
    return 3 * space[0] + space[1]


def print_board(env):
    for row in range(3):
        if row > 0:
            print('-----')
        s = ''
        for col in range(3):
            if env.board[row, col] == 0:
                s += (str(row*3 + col))
            elif env.board[row, col] == 1:
                s += 'X'
            else:
                s += 'O'
            if col < 2:
                s += '|'
        print(s)


# e = TTT2()
# is_finished = False
# while not is_finished:
#     print_board(e.env)
#     user_input = input('\nChoose your move ')
#     output = e.step(int(user_input))
#     is_finished = output[2]
#
# print(output)


