"""
All 9 optimizer configurations, grouped by Part I/II/III.
Each entry: (name, optimizer_builder_fn, batch_size)
"""
from tensorflow.keras.optimizers import SGD, Adagrad, RMSprop, Adam

LR = 0.01          # shared base learning rate for GD-family (Part I & II)
LR_ADAPTIVE = 0.001  # typical lr for adaptive methods (Part III)
EPOCHS = 50
FULL_BATCH = 6249    # ~ size of X_train after val split (set dynamically in main.py)

OPTIMIZER_CONFIGS = {
    # ---------- Part I: Batch-size comparison (plain SGD, no momentum) ----------
    "1_Batch_GD":      {"part": "I",   "batch_size": "full", "build": lambda: SGD(learning_rate=LR)},
    "2_Minibatch_SGD":  {"part": "I",   "batch_size": 32,      "build": lambda: SGD(learning_rate=LR)},
    "3_Stochastic_GD":  {"part": "I",   "batch_size": 1,       "build": lambda: SGD(learning_rate=LR)},

    # ---------- Part II: Momentum variants (fixed mini-batch = 32) ----------
    "4_Gradient_Descent":        {"part": "II", "batch_size": 32, "build": lambda: SGD(learning_rate=LR)},
    "5_GD_Momentum":             {"part": "II", "batch_size": 32, "build": lambda: SGD(learning_rate=LR, momentum=0.9)},
    "6_GD_Nesterov_Momentum":    {"part": "II", "batch_size": 32, "build": lambda: SGD(learning_rate=LR, momentum=0.9, nesterov=True)},

    # ---------- Part III: Adaptive learning-rate methods (fixed mini-batch = 32) ----------
    "7_AdaGrad": {"part": "III", "batch_size": 32, "build": lambda: Adagrad(learning_rate=LR_ADAPTIVE)},
    "8_RMSProp": {"part": "III", "batch_size": 32, "build": lambda: RMSprop(learning_rate=LR_ADAPTIVE)},
    "9_Adam":    {"part": "III", "batch_size": 32, "build": lambda: Adam(learning_rate=LR_ADAPTIVE)},
}