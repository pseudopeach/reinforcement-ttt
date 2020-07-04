import numpy as np
import random
from dqn_model import DQNModel
from ttt_environment import TTT2


base_hparams = {
    'max_mem_size': 20_000,
    'min_mem_size': 1_000,
    'epsilon_decay': 0.9999620984544109,
    'batch_size': 64,
    'min_epsilon': 0.0023,
    'target_model_update_every': 25,
    'evaluation_every': 1000,
    'evaluation_size': 100,
    'beta': 0.987,
}

hparams = base_hparams.copy()

def train(environment, starting_model_path=None, episodes=15000):
    if starting_model_path:
        policy_model = DQNModel.load(starting_model_path)
        target_model = DQNModel.load(starting_model_path)
        print('loaded model {}'.format(starting_model_path))
    else:
        print('starting model from scratch')
        policy_model = DQNModel()
        target_model = DQNModel()
        target_model.set_weights(policy_model.get_weights())

    print('Begin training...')
    replay_memory = []
    epsilon = 0.0

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
            print('===================== episode {}, epsilon {}'.format(episode_i, epsilon))
            print(info)
            print('======================================')
            policy_model.save('checkpoint-{}'.format(episode_i))


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
        verbose=False,
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
        history = play_out_episode(model, environment)
        before_state, action, win_condition, final_state, is_terminal = history[-1]

        assert is_terminal

        record.append((win_condition, len(history)))

    record.sort()

    return {
        'worst_outcome': record[0][0],
        'median_outcome': record[len(record)//2][0],
        'best_outcome': record[-1][0],
        'avg_moves': float(sum(h[1] for h in record)) / len(record)
    }

def get_random_action(environment):
    return random.randrange(environment.get_action_space_size())


# for trial in range(100):
#     hparams = base_hparams.copy()
#     hparams['epsilon_decay'] = 0.99994 + random.random()*.00003
#     hparams['min_epsilon'] = .001 + random.random()*.003
#     hparams['batch_size'] = int(50 + random.random() * 20)
#     hparams['beta'] = 0.99 - random.random() * .1
#     print("\n\n", hparams)
#     train(TTT2())

train(TTT2())
