import numpy as np

import gym

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras import backend as K
from NoisyDense import NoisyDense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# -- constants
ENV = 'CartPole-v0'

EPISODES = 2000
RENDER=False

LOSS_PENALTY=0.5 # Weight to give to loss prediction
ENTROPY_PENALTY=0.001
LOSS_CLIPPING=0.2 # Only implemented clipping for the surrogate loss, paper said it was best
SIGMA_INIT=0.02 # For the noisy net
EPOCHS=10

GAMMA = 0.9

BATCH_SIZE = 32
NUM_ACTIONS=2
NUM_STATE = 4

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1,1))

def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new

def value_loss():
    def val_loss(y_true, y_pred):
        advantage = y_true - y_pred
        return K.mean(LOSS_PENALTY * K.square(advantage))
    return val_loss

def proximal_policy_optimization_loss(actual_value, old_prediction, predicted_value):
    advantage = actual_value - predicted_value
    def loss(y_true, y_pred):
        prob = K.sum(y_pred * y_true, axis=1, keepdims=True) + 1e-10
        old_prob = K.sum(old_prediction * y_true, axis=1, keepdims=True) + 1e-10
        r = prob/old_prob
        entropy = -K.mean(prob * K.log(prob))
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1-LOSS_CLIPPING, max_value=1+LOSS_CLIPPING) * advantage)) + ENTROPY_PENALTY * entropy
    return loss


class Agent:

    def __init__(self):
        self.model = self._build_model()
        self.env = gym.make(ENV)
        self.episode = 0
        self.observation = self.env.reset()
        self.reward = []
        self.reward_over_time = []

    def _build_model(self):

        state_input = Input(shape=(NUM_STATE,))
        actual_value = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(64, activation='relu')(state_input)
        # x = Dropout(0.5)(x)
        # x = Dense(64, activation='relu')(x)
        # x = Dropout(0.5)(x)

        out_value = NoisyDense(1, name='out_value', sigma_init=SIGMA_INIT)(x)
        out_actions = NoisyDense(NUM_ACTIONS, activation='softmax', name='out_actions', sigma_init=SIGMA_INIT)(x)

        model = Model(inputs=[state_input, actual_value, old_prediction], outputs=[out_actions, out_value, actual_value, old_prediction])
        model.compile(optimizer=Adam(),
                      loss=[proximal_policy_optimization_loss(actual_value=actual_value, old_prediction=old_prediction, predicted_value=out_value),
                            value_loss(),
                            'mae',
                            'mae'])

        model.summary()
        return model


    def print_average_weight(self):
        return np.mean(self.model.get_layer('out_actions').get_weights()[1])


    def reset_env(self):
        self.episode += 1
        self.observation = self.env.reset()
        self.reward = []

    def get_action(self):
        p, _, _, _ = self.model.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        action = np.random.choice(NUM_ACTIONS, p=p[0])
        action_matrix = np.zeros((p[0].shape))
        action_matrix[action] = 1

        return action, action_matrix, p


    # All kinds of fucked up, fix it
    def get_reward(self, index, length):
        reward = self.reward[-length + index]
        # reward = -self.reward[-length + index] + self.reward[min(-length + index + TIMESTEP, len(self.reward)-1)]
        return reward
    # All kinds of fucked up, fix it
    def transform_reward(self):
        if self.episode % 100 == 0:
            print('Episode # ', self.episode, 'finished with reward', np.array(self.reward).sum())
            print('Average Random Weights',self.print_average_weight())
        self.reward_over_time.append(np.array(self.reward).sum())
        for j in range(len(self.reward)):
            reward = self.reward[j]
            for k in range(j + 1, len(self.reward)):
                reward += self.reward[k] * GAMMA ** k
            self.reward[j] = reward


    def get_batch(self):
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[0]) < BATCH_SIZE:
            action, action_matrix, predicted_action = self.get_action()
            observation, reward, done, info = self.env.step(action)
            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                for i in range(len(tmp_batch[0])):
                    obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                    r = self.get_reward(i, len(tmp_batch[0]))
                    batch[0].append(obs)
                    batch[1].append(action)
                    batch[2].append(pred)
                    batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        while self.episode < EPISODES:
            obs, action, pred, reward = self.get_batch()
            old_prediction = pred
            for e in range(EPOCHS):
                self.model.train_on_batch([obs, reward, old_prediction], [action, reward, reward, old_prediction])
            self.model.get_layer('out_actions').sample_noise()
            self.model.get_layer('out_value').sample_noise()

if __name__ == '__main__':
    ag = Agent()
    ag.run()
    old, ma_list = 0, []
    for value in ag.reward_over_time:
        old = exponential_average(old, value, .99)
        ma_list.append(old)

    plt.plot(ma_list)
    plt.show()