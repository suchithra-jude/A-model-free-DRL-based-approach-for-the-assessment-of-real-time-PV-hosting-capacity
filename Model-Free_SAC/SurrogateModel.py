import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from networks import SurrogateNetwork


class SurrogateAgent:
    def __init__(self,  alpha=0.001, n_actions=10):
        self.Surrogate = SurrogateNetwork(n_actions=n_actions, name='Surrogate')
        self.Surrogate.compile(optimizer=Adam(learning_rate=alpha), loss='mean_squared_error')

    def learn(self, states, target):
        with tf.GradientTape() as tape:
            target = tf.convert_to_tensor(target, dtype=tf.float32)
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = self.Surrogate(states)
            actions = 0.1 + (0.2 * (actions + 1) / 2)
            Surrogate_loss = keras.losses.MSE(target, actions)

        Surrogate_gradient = tape.gradient(Surrogate_loss,
                                          self.Surrogate.trainable_variables)
        self.Surrogate.optimizer.apply_gradients(
                   zip(Surrogate_gradient, self.Surrogate.trainable_variables))
        return Surrogate_loss.numpy()


    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mu_prime = self.Surrogate(state)[0]
        mu_prime = 0.1 + (0.2 * (mu_prime + 1) / 2)
        return mu_prime.numpy()


    def save_models(self):
        print('... saving models ...')
        self.Surrogate.save_weights(self.Surrogate.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.Surrogate.load_weights(self.Surrogate.checkpoint_file)

