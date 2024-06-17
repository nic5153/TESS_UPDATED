import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from numpy.polynomial import Polynomial

class LightCurveWithConstraint:
    def __init__(self, filename, x_min=None, x_max=None, cadence_days=0.00277778, period=None):
        self.filename = filename
        self.x_min = x_min
        self.x_max = x_max
        self.cadence_days = cadence_days
        self.period = period

        self.data = self._read_data()
        self.detrended_flux = None

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

    def _compute_aic(self, flux, trend, k):
        resid = flux - trend
        sse = np.sum(resid**2)
        aic = 2 * k + len(flux) * np.log(sse / len(flux))
        return aic

    def detrend(self, max_degree=5):
        t = self.data[:, 1]
        flux = self.data[:, 4]

        best_aic = np.inf
        best_degree = 0
        best_trend = None

        for degree in range(1, max_degree + 1):
            p = Polynomial.fit(t, flux, deg=degree)
            trend = p(t)
            aic = self._compute_aic(flux, trend, degree + 1)

            if aic < best_aic:
                best_aic = aic
                best_degree = degree
                best_trend = trend

        self.detrended_flux = flux - best_trend
        print(f"Selected polynomial degree: {best_degree}")

    def lomb_scargle(self):
        if self.detrended_flux is None:
            raise ValueError("Please detrend the data before applying Lomb-Scargle periodogram.")
        
        t = np.diff(self.data[:, 1])
        max_frequency = 0.5 / np.mean(t)
        frequency = np.linspace(0.01, max_frequency, 10000)
    
        ls = LombScargle(self.data[:, 1], self.detrended_flux)
        power = ls.power(frequency)
        period_days = 1 / frequency

        peaks, _ = find_peaks(power, height=0.20)
        print("Significant peaks:")
        for peak_index in peaks:
            peak_period = period_days[peak_index]
            peak_power = power[peak_index]
            print(f"Period: {peak_period:.5f} days, Power: {peak_power:.5f}")
        return frequency, power, period_days

    def phasefold(self):
        if self.period is None:
            raise ValueError("Period must be provided for phase folding.")
        t = self.data[:, 1]
        p = self.period
        self.phi = (t % p) / p  
        sorted_indices = np.argsort(self.phi)
        return self.phi[sorted_indices]

    def plot(self, save_dir=None):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # Plot original light curve
        axs[0].plot(self.data[:, 1], self.data[:, 4], color='g', label="Original Light Curve")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Magnitude")
        axs[0].set_title("Original Light Curve")
        axs[0].invert_yaxis()
        axs[0].legend()

        
        axs[1].plot(self.data[:, 1], self.detrended_flux, color='r', label="Detrended Light Curve")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Detrended Magnitude")
        axs[1].set_title("Detrended Light Curve")
        axs[1].invert_yaxis()
        axs[1].legend()

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "light_curve.png"))
        else:
            plt.show()

        
        frequency, power, period_days = self.lomb_scargle()
        plt.figure()
        plt.plot(period_days, power, color='b', label="LS Periodogram")
        plt.xlabel("Period (days)")
        plt.ylabel("Power")
        plt.title("LS Periodogram")
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "periodogram.png"))
        else:
            plt.show()
        
        
        if self.period is not None:
            plt.figure()
            phi_sorted = self.phasefold()
            sorted_indices = np.argsort(self.phi)
            plt.scatter(phi_sorted, self.detrended_flux[sorted_indices], color='m', s=5, label="Phase Folded Curve")
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

