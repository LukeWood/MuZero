import math

import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import Regularizer

from game.game import Action
from networks.network import BaseNetwork


class BreakoutNetwork(BaseNetwork):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 representation_size: int,
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-4,
                 representation_activation: str = 'tanh'):
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1

        regularizer = regularizers.l2(weight_decay)

        representation_network = self._get_representation_network(
            representation_size, regularizer)

        value_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                    Dense(self.value_support_size, kernel_regularizer=regularizer)])
        policy_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer), Dense(
            action_size, kernel_regularizer=regularizer)])
        dynamic_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                      Dense(representation_size, activation=representation_activation,
                                            kernel_regularizer=regularizer)])
        reward_network = Sequential([Dense(16, activation='relu', kernel_regularizer=regularizer),
                                     Dense(1, kernel_regularizer=regularizer)])

        super().__init__(representation_network, value_network,
                         policy_network, dynamic_network, reward_network)

    def _get_representation_network(self, representation_size: int,
                                    regularizer: Regularizer) -> Sequential:
        """ Constructs the representation network for breakout """
        model = Sequential()
        model.add(Conv2D(32, 3, strides=(1, 1),
                         padding="same", activation="relu", data_format="channels_last"))
        model.add(Conv2D(64, 4, strides=(2, 2),
                         padding="same", activation="relu", data_format="channels_last"))
        model.add(Flatten())
        model.add(Dense(256, activation='relu',
                        kernel_regularizer=regularizer))
        # TODO change tanh to activation used in the paper minmax
        model.add(Dense(representation_size, activation='tanh',
                        kernel_regularizer=regularizer))
        return model

    def _value_transform(self, value_support: np.array) -> float:
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """

        value = self._softmax(value_support)
        value = np.dot(value, range(self.value_support_size))
        value = np.asscalar(value) ** 2
        return value

    def _reward_transform(self, reward: np.array) -> float:
        return np.asscalar(reward)

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        conditioned_hidden = np.concatenate(
            (hidden_state, np.eye(self.action_size)[action.index]))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)
