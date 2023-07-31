import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=512, fc3_dims=1024, fc4_dims=512, fc5_dims=256,
            name='critic', chkpt_dir='tmp/sac'):

        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(self.fc3_dims, activation='relu')
        self.fc4 = Dense(self.fc4_dims, activation='relu')
        self.fc5 = Dense(self.fc5_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        action_value = self.fc3(action_value)
        action_value = self.fc4(action_value)
        action_value = self.fc5(action_value)

        q = self.q(action_value)

        return q


class ActorNetwork(keras.Model):
    def __init__(self, max_action=1, fc1_dims=256, fc2_dims=512, fc3_dims=1024, fc4_dims=512, fc5_dims=256,
            n_actions=2, name='actor', chkpt_dir='tmp/sac'):

        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.noise = 1e-6

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(self.fc3_dims, activation='relu')
        self.fc4 = Dense(self.fc4_dims, activation='relu')
        self.fc5 = Dense(self.fc5_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        prob = self.fc3(prob)
        prob = self.fc4(prob)
        prob = self.fc5(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.call(state)
        #sigma = 1e-8
        probabilities = tfp.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.sample()
        else:
            actions = probabilities.sample()

        action = tf.math.tanh(actions)
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        action = action * self.max_action

        return action, log_probs


class SurrogateNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=512, fc3_dims=1024, fc4_dims=512, fc5_dims=256, n_actions=10, name='Surrogate',
            chkpt_dir='tmp/Surrogate'):

        super(SurrogateNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_TD3.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(self.fc3_dims, activation='relu')
        self.fc4 = Dense(self.fc4_dims, activation='relu')
        self.fc5 = Dense(self.fc5_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        prob = self.fc3(prob)
        prob = self.fc4(prob)
        prob = self.fc5(prob)
        mu = self.mu(prob)

        return mu