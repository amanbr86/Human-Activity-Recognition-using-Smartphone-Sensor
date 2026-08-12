"""
Defines the SINGLE fixed feed-forward architecture used for all 9 runs.
Keeping architecture + init identical isolates the optimizer as the only
independent variable.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.initializers import GlorotUniform

SEED = 42
INPUT_DIM = 561
NUM_CLASSES = 6


def build_model():
    init = GlorotUniform(seed=SEED)
    model = Sequential([
        Input(shape=(INPUT_DIM,)),
        Dense(128, activation="relu", kernel_initializer=init),
        Dropout(0.3),
        Dense(64, activation="relu", kernel_initializer=init),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax", kernel_initializer=init),
    ])
    return model