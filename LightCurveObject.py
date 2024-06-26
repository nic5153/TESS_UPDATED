import os
import numpy as np
import matplotlib.pyplot as plt

class LightCurve:
    def __init__(self, filename, target=None):
        self.filename = filename
        self.data = self._read_data()
        self.target = target

    def _read_data(self):
        return np.loadtxt(self.filename)

    def plot(self):
        plt.plot(self.data[:, 1], self.data[:, 4], label="Light Curve")
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.title("Light Curve")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()

    def save_plots(self, save_dir=None):
        if save_dir is None:
            save_dir = "./"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.figure()
        plt.plot(self.data[:, 1], self.data[:, 4], label="Light Curve")
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.title("Light Curve")
        plt.legend()
        plt.gca().invert_yaxis()
        if self.target is not None:
            plt.savefig(os.path.join(save_dir, f"light_curve_{self.target}.png"))
        else:
            plt.savefig(os.path.join(save_dir, "light_curve.png"))
        plt.close()


