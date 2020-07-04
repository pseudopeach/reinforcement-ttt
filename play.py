from ttt_environment import TTT
from dqn_model import DQNModel


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

e = TTT()
e.reset()

model = DQNModel.load('awesome2.pb')

is_finished = False
output = None
winner = None
while not is_finished:
    print_board(e)
    user_input = input('\nChoose your move ')
    output = e.step(int(user_input))

    if not is_finished:
        predicted_qs = model.predict(e.board)[0]
        for index, q in enumerate(predicted_qs):
            print('{}: {}'.format(index, q))
        ai_action = model.get_top_action(e.board)
        output = e.step(ai_action)
        is_finished = output[2]

print('final output', output)
