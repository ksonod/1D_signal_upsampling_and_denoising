import tensorflow as tf
from model import build_model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

INPUT_FILES = {
    "X": "X.npy",  # Noisy low-sampled data
    "y": "y.npy",  # Noise-less high-resolution data
    # "model_weights": "model1_64to256.h5",  # It can be commented out.
}

CONFIG = {
    # Total data is split into training and test data with this ratio.
    "test_data_size": 0.2,

    # This ratio is based on training data. In other words, test data size is not considered.
    # For example, if this parameter is set to be 0.2, it means the training and validation ratio is 0.8:0.2.
    "validation_size": 0.2,

    "model_params": {
        "input_shape": (128, 1),  # (256, 1). It should be the same as the filters.
        "num_residual_blocks": 16,  # 32
        "filters": 128,  # 256. It should be the same as the input shape.
        "scaling_factor": 4,  # 4
    },
    "training_params": {
        "epochs": 10,
        "batch_size": 64
    },
    "model_optimizer": {  # adadelta, adafactor
        "optimizer": tf.keras.optimizers.Adam(
            # learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            #     2e-4, decay_steps=200, decay_rate=0.95, staircase=True
            # ),  # 0.001
            # beta_1=0.9,  # 0.9
            # beta_2=0.999,  # 0.999
        ),
        "loss": tf.keras.losses.MeanAbsoluteError(),
    },
    "callbacks": [
        tf.keras.callbacks.ModelCheckpoint(**{
            "filepath": "model.h5",
            "monitor": "val_loss",
            "verbose": 1,
            "save_best_only": True,
            "save_weights_only": False,
            "save_freq": "epoch",
        }),
        tf.keras.callbacks.EarlyStopping(**{
            "monitor": "val_loss",
            "patience": 8,
            "verbose": 1,
            "restore_best_weights": True,
        }),
    ],
    "train_model": True,
    "evaluate_on_test_data": True,
}


def evaluate_model_on_test_data(model, X_test, y_test):
    y_pred = model.predict(X_test)
    test_mse = tf.keras.losses.MSE(y_test, np.squeeze(y_pred))._numpy().mean()
    print(f"Results for the test data: {test_mse}")

    scaling_factor = int(y_test.shape[1] / X_test.shape[1])
    num_data_to_visualize = 5
    plt.figure(figsize=(4, 8))
    for i in range(num_data_to_visualize):
        plt.subplot(num_data_to_visualize, 1, i + 1)
        data_idx = np.random.randint(low=0, high=y_pred.shape[0])
        plt.plot(
            np.arange(0, y_test.shape[1], 1)[::scaling_factor], X_test[data_idx, :].flatten(),
            "b-", alpha=0.5, label="input"
        )
        plt.plot(
            np.arange(0, y_test.shape[1], 1), y_pred[data_idx, :].flatten(),
            "r-", alpha=0.5, label="prediction"
        )
        plt.plot(
            np.arange(0, y_test.shape[1], 1), y_test[data_idx, :].flatten(),
            "k--", label="ground-truth"
        )
        plt.legend()
        plt.tight_layout()
    plt.savefig("prediction_examples.png")


def main(input_files, config):
    X = np.load(input_files["X"])
    y = np.load(input_files["y"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_data_size"])

    model = build_model(
        input_shape=config["model_params"]["input_shape"],
        num_residual_blocks=config["model_params"]["num_residual_blocks"]
    )
    model.summary()
    model.compile(**config["model_optimizer"])
    # model.compile(optimizer="adam", loss="mse")

    if "model_weights" in input_files:
        model.load_weights(input_files["model_weights"])

    # Train a model
    if config["train_model"]:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=config["training_params"]["batch_size"],
            epochs=config["training_params"]["epochs"],
            validation_split=config["validation_size"],
            shuffle=True,
            callbacks=config["callbacks"]
        )

        with open("training_history.json", "w") as f:
            json.dump(history.history, f)

        plt.figure()
        plt.plot(history.epoch, history.history["loss"], label="train")
        plt.plot(history.epoch, history.history["val_loss"], label="val")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig("training_curve.png")

    if config["evaluate_on_test_data"]:
        evaluate_model_on_test_data(model, X_test, y_test)
    plt.show()


if __name__ == "__main__":
    main(input_files=INPUT_FILES, config=CONFIG)



