import numpy as np
import random
from dqn_model import DQNModel
from ttt_environment import TTT2


hparams = {
    'max_mem_size': 20_000,
    'min_mem_size': 1_000,
    'epsilon_decay': 0.999,
    'batch_size': 64,
    'min_epsilon': 0.001,
    'target_model_update_every': 25,
    'evaluation_every': 25,
    'evaluation_size': 100,
    'beta': 0.99,
}


def train(environment, starting_model_path=None, episodes=1000):
    if starting_model_path:
        policy_model = DQNModel.load(starting_model_path)
        target_model = DQNModel.load(starting_model_path)
    else:
        policy_model = DQNModel()
        target_model = DQNModel()
        target_model.set_weights(policy_model.get_weights())

    print('Begin training...')
    replay_memory = []
    epsilon = 1.0

    for episode_i in range(episodes):
        replay_memory += play_out_episode(policy_model, environment, epsilon)
        replay_memory = replay_memory[-hparams['max_mem_size']:]

        epsilon = max(hparams['min_epsilon'], epsilon*hparams['epsilon_decay'])
        if len(replay_memory) >= hparams['min_mem_size']:
            do_training_step(policy_model, target_model, random.sample(replay_memory, hparams['batch_size']))

        if episode_i % hparams['target_model_update_every'] == 0:
            target_model.set_weights(policy_model.get_weights())
        if episode_i % hparams['evaluation_every'] == 0:
            info = evaluate_model(policy_model, environment)
            print('===================== episode {}'.format(episode_i))
            print(info)
            print('======================================')
            policy_model.save('checkpoint-{}')


def do_training_step(policy_model, target_model, batch):

    initial_states = np.array([transition[0] for transition in batch])
    targets = target_model.predict(initial_states)

    new_states = np.array([transition[3] for transition in batch])
    new_q_estimates = target_model.predict(new_states)

    for index, (state, action, reward, new_state, is_terminal) in enumerate(batch):
        q = reward
        if not is_terminal:
            q += hparams['beta']*np.max(new_q_estimates[index])

        targets[index][action] = q

    policy_model.fit(
        initial_states,
        targets,
        batch_size=hparams['batch_size'],
        shuffle=False,
    )


def play_out_episode(policy_model, environment, epsilon=0.0):
    history = []
    current_state = environment.reset()

    while True:
        if np.random.random() > epsilon:
            action = policy_model.get_top_action(current_state)
        else:
            action = get_random_action(environment)

        new_state, reward, is_terminal = environment.step(action)

        history.append((current_state, action, reward, new_state, is_terminal))

        current_state = new_state

        if is_terminal: break

    return history


def evaluate_model(model, environment):
    record = []
    for episode_i in range(hparams['evaluation_size']):
        environment.reset()
        before_state, action, win_condition, final_state, is_terminal = play_out_episode(model, environment)[-1]

        assert is_terminal

        record.append(win_condition)

    record.sort()

    return {
        'worst_outcome': record[0],
        'median_outcome': record[len(record)//2],
        'best_outcome': record[-1],
    }

def get_random_action(environment):
    return random.randrange(environment.get_action_space_size())


train(TTT2())
