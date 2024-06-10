import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

class LightCurveWithConstraint:
    def __init__(self, filename, x_min=None, x_max=None, cadence_minutes=2, period=None):
        self.filename = filename
        self.x_min = x_min
        self.x_max = x_max
        self.cadence_minutes = cadence_minutes
        self.period = period

        self.data = self._read_data()

    def _read_data(self):
        data = np.loadtxt(self.filename)
        if self.x_min is not None and self.x_max is not None:
            mask = (data[:, 1] >= self.x_min) & (data[:, 1] <= self.x_max)
            return data[mask]
        elif self.x_min is not None:
            return data[data[:, 1] >= self.x_min]
        elif self.x_max is not None:
            return data[data[:, 1] <= self.x_max]
        else:
            return data

    def lomb_scargle(self):
        max_frequency = 1 / (self.cadence_minutes / 60)
        frequency = np.linspace(0.5, max_frequency, 10000)

        ls = LombScargle(self.data[:, 1], self.data[:, 4])
        power = ls.power(frequency)
        period_hours = 24 / frequency
        peaks, _ = find_peaks(power, height=0.20)
        print("Significant peaks:")
        for peak_index in peaks:
            peak_period = period_hours[peak_index]
            peak_power = power[peak_index]
            print(f"Period: {peak_period:.2f} hours, Power: {peak_power:.2f}")
        return frequency, power, period_hours

    def phasefold(self):
        t = self.data[:, 1]
        p = self.period
        self.phi = (t/p)%1
        return self.phi


    def plot(self, save_dir=None):
        frequency, power, period_hours = self.lomb_scargle()

        plt.figure()
        plt.plot(self.data[:, 1], self.data[:, 4], label="Light Curve W/ Constraint")
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.title("Light Curve W/ Constraint")
        plt.legend()
        plt.gca().invert_yaxis()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "light_curve.png"))
        else:
            plt.show()

        plt.figure()
        plt.plot(period_hours, power, label="LS Periodogram W/ Constraint")
        plt.xlabel("Period (hours)")
        plt.ylabel("Power")
        plt.title("LS Periodogram W/ Constraint")
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "periodogram.png"))
        else:
            plt.show()
        
        plt.figure()
        self.phasefold()
        sorted_indices = np.argsort(self.phi)
        plt.plot(self.phi[sorted_indices], self.data[:, 4][sorted_indices], color='r', label="Phase Folded Curve")
        plt.xlabel("Phase")
        plt.ylabel("Magnitude")
        plt.title("Phase Folded Curve")
        plt.gca().invert_yaxis()
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "phase_folded_curve.png"))
        else:
            plt.show()

    def save_plots(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.plot(save_dir)