import numpy as np

import gym

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numba

# -- constants
ENV = 'Breakout-ram-v0'

EPISODES = 5000
RENDER=False

ENTROPY_PENALTY=0.005
LOSS_CLIPPING=0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS=10

GAMMA = 0.95

BATCH_SIZE = 64
NUM_ACTIONS = 4
NUM_STATE = 128

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1,1))

def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new

def proximal_policy_optimization_loss(actual_value, predicted_value, old_prediction):
    advantage = actual_value - predicted_value
    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=1, keepdims=True)
        old_prob = K.sum(y_true * old_prediction, axis=1, keepdims=True)

        r = prob/(old_prob + 1e-10)

        return -K.log(prob + 1e-10) * K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage))
    return loss

class Agent:

    def __init__(self):
        self.critic = self.build_critic()
        self.actor = self.build_actor()
        self.env = gym.make(ENV)
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.observation = self.env.reset()
        self.reward = []
        self.reward_over_time = []

    def build_actor(self):

        state_input = Input(shape=(NUM_STATE,))
        actual_value = Input(shape=(1,))
        predicted_value = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(64, activation='relu')(state_input)

        out_actions = Dense(NUM_ACTIONS, activation='softmax')(x)

        model = Model(inputs=[state_input, actual_value, predicted_value, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(),
                      loss=[proximal_policy_optimization_loss(
                          actual_value=actual_value,
                          old_prediction=old_prediction,
                          predicted_value=predicted_value)])

        return model

    def build_critic(self):

        state_input = Input(shape=(NUM_STATE,))
        x = Dense(64, activation='relu')(state_input)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(), loss='mse')

        return model

    @numba.jit
    def reset_env(self):
        self.episode += 1
        self.observation = self.env.reset()
        self.reward = []

    @numba.jit
    def get_action(self):
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_VALUE, DUMMY_ACTION])
        action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        action_matrix = np.zeros((p[0].shape))
        action_matrix[action] = 1

        return action, action_matrix, p


    def transform_reward(self):
        if self.episode % 100 == 0:
            print('Episode # ', self.episode, 'finished with reward', np.array(self.reward).sum())
            # print('Average Random Weights',self.print_average_weight())
            # self.actor.save_weights('actor')
            # self.critic.save_weights('critic')
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
            pred_values = self.critic.predict(obs)
            for e in range(EPOCHS):
                self.actor.train_on_batch([obs, reward, pred_values, old_prediction], [action])
            for e in range(EPOCHS):
                self.critic.train_on_batch([obs], [reward])

if __name__ == '__main__':
    ag = Agent()
    ag.run()
    old, ma_list = 0, []
    for value in ag.reward_over_time:
        old = exponential_average(old, value, .99)
        ma_list.append(old)

    plt.plot(ma_list)
    plt.show()