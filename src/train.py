"""
Generic trainer: builds a fresh model, compiles with the given optimizer,
trains, evaluates, and returns all metrics + history.
"""
import time
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

from model import build_model, SEED
import tensorflow as tf


def train_and_evaluate(name, optimizer_builder, batch_size,
                        X_train, y_train, X_val, y_val,
                        X_test, y_test, y_test_raw, epochs=50):

    tf.random.set_seed(SEED)  # identical weight init across all 9 runs
    model = build_model()
    optimizer = optimizer_builder()

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    bs = X_train.shape[0] if batch_size == "full" else batch_size

    early_stop = EarlyStopping(monitor="val_loss", patience=8,
                                restore_best_weights=True)

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=bs,
        callbacks=[early_stop],
        verbose=0
    )
    train_time = time.time() - start

    # Evaluate on test set
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = accuracy_score(y_test_raw, y_pred)
    test_f1 = f1_score(y_test_raw, y_pred, average="macro")
    cm = confusion_matrix(y_test_raw, y_pred)

    result = {
        "name": name,
        "batch_size": bs,
        "epochs_run": len(history.history["loss"]),
        "train_time_sec": round(train_time, 2),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
        "final_train_loss": round(history.history["loss"][-1], 4),
        "final_val_loss": round(history.history["val_loss"][-1], 4),
    }

    # Save history for plotting later
    with open(f"results/history/{name}.json", "w") as f:
        json.dump(history.history, f)

    return result, cm, history