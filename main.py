import tensorflow as tf
from model import build_model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

INPUT_FILES = {
    "X": "X.npy",  # Noisy low-sampled data
    "y": "y.npy",  # Noise-less high-resolution data
}

CONFIG = {
    # Total data is split into training and test data with this ratio.
    "test_data_size": 0.2,

    # This ratio is based on training data. In other words, test data size is not considered.
    # For example, if this parameter is set to be 0.2, it means the training and validation ratio is 0.8:0.2.
    "validation_size": 0.2,

    "model_params": {
        "input_shape": (256, 1),
        "num_residual_blocks": 10,  # 32
        "filters": 256,  # 256
        "scaling_factor": 4,  # 4
    },
    "model_optimizer": {
        "optimizer": tf.keras.optimizers.Adam(
            learning_rate=0.001,  # 0.001
            beta_1=0.9,  # 0.9
            beta_2=0.999,  # 0.999
        ),
        "loss": tf.keras.losses.MeanSquaredError(),
    },
    "training_params": {
        "epochs": 10,
        "batch_size": 32
    },
    "evaluate_on_test_data": True,
}

def main(input_files, config):
    X = np.load(input_files["X"])
    y = np.load(input_files["y"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_data_size"])

    model = build_model(
        input_shape=config["model_params"]["input_shape"],
        num_residual_blocks=config["model_params"]["num_residual_blocks"]
    )
    model.summary()
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        x=X_train[:5000],
        y=y_train[:5000],
        batch_size=config["training_params"]["batch_size"],
        epochs=config["training_params"]["epochs"],
        validation_split=config["validation_size"],
        shuffle=True
    )

    plt.figure()
    plt.plot(history.epoch, history.history["loss"], ".", label="train")
    plt.plot(history.epoch, history.history["val_loss"], ".", label="val")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    if config["evaluate_on_test_data"]:
        y_pred = model.predict(X_test)
        test_mse = tf.keras.losses.MSE(y_test, np.squeeze(y_pred))._numpy().mean()
        print(f"Results for the test data: {test_mse}")

        scaling_factor = int(y.shape[1]/X.shape[1])
        num_data_to_visualize = 5
        plt.figure()
        for i in range(num_data_to_visualize):
            plt.subplot(num_data_to_visualize, 1, i+1)
            data_idx = np.random.randint(low=0, high=y_pred.shape[0])
            plt.plot(
                np.arange(0, y.shape[1], 1)[::scaling_factor], X_test[data_idx, :].flatten(),
                "b-", alpha=0.5, label="input"
            )
            plt.plot(
                np.arange(0, y.shape[1], 1), y_pred[data_idx, :].flatten(),
                "r-", alpha=0.5, label="prediction"
            )
            plt.plot(
                np.arange(0, y.shape[1], 1), y_test[data_idx, :].flatten(),
                "k--", label="ground-truth"
            )
            plt.legend()
    plt.show()


if __name__ == "__main__":
    main(input_files=INPUT_FILES, config=CONFIG)



