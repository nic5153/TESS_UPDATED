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

    def plot(self, save_dir=None):
        plt.figure()
        plt.plot(self.data[:, 1], self.data[:, 4], label="Light Curve")
        plt.xlabel("Time (BTJD)")
        plt.ylabel("Magnitude")
        plt.title(f"Light Curve of {self.target} ")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.gcf().set_size_inches(18.5, 10.5)

        if save_dir:
            if self.target is not None:
                plt.savefig(os.path.join(save_dir, f"light_curve_{self.target}.png"))
            else:
                plt.savefig(os.path.join(save_dir, "light_curve.png"))
        else:
            plt.show()
        plt.close()

    def save_plots(self, save_dir):
        self.plot(save_dir)


