import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

class LightCurve:
    def __init__(self, filename):
        self.filename = filename
        self.data = self._read_data()

    def _read_data(self):
        return np.loadtxt(self.filename)
        
    def LombScargle(self):
        cadence_minutes = 2
        max_frequency = 1 / (cadence_minutes / 60)
        frequency = np.linspace(0.05, max_frequency, 1000000)
        ls = LombScargle(self.data[:, 1], self.data[:, 4])
        power = ls.power(frequency)
        period_hours = 24 / frequency
        peaks, _ = find_peaks(power, height=.5)
        print("Significant peaks:")
        for peak_index in peaks:
            peak_period = period_hours[peak_index]
            peak_power = power[peak_index]
            print(f"Period: {peak_period:.2f} hours, Power: {peak_power:.2f}")
        return frequency, power, period_hours
    
    def plot(self):
        plt.subplot(1,2,1)
        plt.plot(self.data[:, 1], self.data[:, 4], label="Light Curve")
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.title("Light Curve")
        plt.legend()
        
        
        frequency, power, period_hours = self.LombScargle()
        plt.subplot(1,2,2)
        plt.plot(period_hours, power, label="Lomb-Scargle Periodogram")
        plt.xlabel("Period")
        plt.ylabel("Power")
        plt.title("Lomb-Scargle Periodogram")
        plt.legend()
        
        plt.show()