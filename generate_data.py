import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

CONFIG = {
    "num_total_data": 50000,
    "x": np.arange(0, 512, 1),  # np.arange(0, 1024, 1)
    "x_upsampling_factor": 4,  # 4
    "center_range_factor": 0.25,
    "num_peaks": [2, 5],  # [minimum, maximum]
    "peak_width_range": [10, 15],  # [minimum width, maximum width]
    "amplitude_range": [0.5, 2],  # 1 is a reference. [minimum amplitude, maximum amplitude]
    "noise_scale": [0.01, 0.05],  # [minimum, maximum]. Scale is used in numpy.random.normal
    "clip_at_0": True,  # True means the minimum signal value will be 0 (i.e., no negative number).
    "show_example": True,
}


def gaussian(x, width, center, amplitude):
    """
    Parametric form of a Gaussian function
    """
    gaussian_signal = amplitude * np.exp(- 0.5 * (x-center)**2 / width ** 2)
    return gaussian_signal.astype(np.float16)


def normalize_signal(signal):
    signal -= signal.min()
    return signal/signal.max()


def generate_signal(config):

    num_peaks = np.random.randint(low=config["num_peaks"][0], high=config["num_peaks"][1])

    signal_dict = {
        "x": config["x"],
        "signal": np.zeros_like(config["x"]).astype(np.float16),
        "width_list": [],
        "center_list": [],
        "amplitude_list": []
    }

    center_range = [
        config["x"].max()*config["center_range_factor"],
        (1 - config["center_range_factor"]) * config["x"].max()
    ]
    for peak_idx in range(num_peaks):
        signal_dict["width_list"].append(
            np.random.uniform(low=config["peak_width_range"][0], high=config["peak_width_range"][1])
        )
        signal_dict["center_list"].append(
            np.random.uniform(low=center_range[0], high=center_range[1])
        )
        signal_dict["amplitude_list"].append(
            np.random.uniform(low=config["amplitude_range"][0], high=config["amplitude_range"][1])
        )

        signal_dict["signal"] += gaussian(
            x=config["x"],
            width=signal_dict["width_list"][-1],
            center=signal_dict["center_list"][-1],
            amplitude=signal_dict["amplitude_list"][-1]
        )

    signal_dict["signal"] = normalize_signal(signal_dict["signal"])

    noise_scale = np.random.uniform(low=config["noise_scale"][0], high=config["noise_scale"][1])
    random_noise = np.random.normal(loc=0, scale=noise_scale, size=config["x"].shape[0])
    signal_dict["noisy_signal"] = signal_dict["signal"] + random_noise
    if config["clip_at_0"]:
        signal_dict["noisy_signal"][signal_dict["noisy_signal"] <= 0] = 0
    return signal_dict


def main(config):

    X = []
    y = []

    for i in tqdm(range(config["num_total_data"])):
        signal_dict = generate_signal(config)

        # Down sampling signals
        for key in ["noisy_signal", "signal", "x"]:
            signal_dict[f"down_sampled_{key}"] = signal_dict[key][::config["x_upsampling_factor"]]

        X.append(signal_dict["down_sampled_noisy_signal"])
        y.append(signal_dict["signal"])

    X = np.array(X)
    y = np.array(y)

    # Save data
    print(f"X shape: {X.shape}. y shape {y.shape}")
    np.save(arr=X, file="X.npy")
    np.save(arr=y, file="y.npy")

    if config["show_example"]:  # The last data will be shown.
        plt.figure()
        plt.plot(signal_dict["x"], signal_dict["noisy_signal"], "r-", alpha=0.5)
        plt.plot(signal_dict["x"], signal_dict["signal"], "k-")
        plt.plot(signal_dict["down_sampled_x"], signal_dict["down_sampled_noisy_signal"], "b.-", alpha=0.6, markersize=2)
        plt.show()


if __name__ == "__main__":
    main(config=CONFIG)



