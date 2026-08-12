"""
Loads and preprocesses the UCI HAR Dataset.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Resolve path relative to THIS file's location, not the current working directory.
# src/data_loader.py -> go up one level -> project root -> data/UCI HAR Dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "UCI HAR Dataset")


def load_data(val_split=0.15, random_state=42):
    """
    Returns X_train, X_val, X_test, y_train, y_val, y_test (one-hot),
    plus y_test_labels (integer, for confusion matrix / F1).
    """
    X_train = pd.read_csv(f"{DATA_DIR}/train/X_train.txt",
                           sep=r"\s+", header=None).values
    y_train_raw = pd.read_csv(f"{DATA_DIR}/train/y_train.txt",
                               header=None).values.ravel() - 1  # 0-indexed

    X_test = pd.read_csv(f"{DATA_DIR}/test/X_test.txt",
                          sep=r"\s+", header=None).values
    y_test_raw = pd.read_csv(f"{DATA_DIR}/test/y_test.txt",
                              header=None).values.ravel() - 1

    # Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train/validation split
    X_train, X_val, y_train_raw, y_val_raw = train_test_split(
        X_train, y_train_raw,
        test_size=val_split,
        random_state=random_state,
        stratify=y_train_raw
    )

    num_classes = 6
    y_train = to_categorical(y_train_raw, num_classes)
    y_val = to_categorical(y_val_raw, num_classes)
    y_test = to_categorical(y_test_raw, num_classes)

    return X_train, X_val, X_test, y_train, y_val, y_test, y_test_raw


def get_activity_labels():
    labels = pd.read_csv(f"{DATA_DIR}/activity_labels.txt",
                          sep=r"\s+", header=None, index_col=0)
    return labels[1].tolist()


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, y_test_raw = load_data()
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)
    print("Classes:", get_activity_labels())