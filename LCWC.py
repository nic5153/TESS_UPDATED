import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from numpy.polynomial import Polynomial

class LightCurveWithConstraint:
    def __init__(self, filename, x_min=None, x_max=None, cadence_days=0.00277778, period=None, power=None, target=None, peak_threshold=None):
        self.filename = filename
        self.x_min = x_min
        self.x_max = x_max
        self.cadence_days = cadence_days
        self.period = period
        self.power = power
        self.target = target
        self.peak_threshold = peak_threshold

        self.data = self._read_data()
        self.detrended_flux = None

    def _read_data(self):
        try:
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
        except Exception as e:
            print(f"Error reading data: {e}")
            return np.loadtxt(self.filename, skiprows=1)

    def _compute_aic(self, flux, trend, k):
        resid = flux - trend
        sse = np.sum(resid**2)
        aic = 2 * k + len(flux) * np.log(sse / len(flux))
        return aic

    def detrend(self, max_degree=5):
        if max_degree == 0:
            self.detrended_flux = self.data[:, 4]
            self.detrended_flux -= np.mean(self.detrended_flux)
            return

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
        
        mean_flux = np.mean(flux)
        self.detrended_flux -= np.mean(self.detrended_flux) - mean_flux

        if np.mean(self.detrended_flux) > mean_flux:
            self.detrended_flux = mean_flux - self.detrended_flux
            
    def lomb_scargle(self, filter_alias=True):
        if self.detrended_flux is None:
            raise ValueError("Please detrend the data before applying Lomb-Scargle periodogram.")
        
        t = self.data[:, 1]
        max_frequency = 0.5 / np.mean(np.diff(t))
        frequency = np.linspace(0.0003, max_frequency, 10000)
        
        ls = LombScargle(t, self.detrended_flux)
        power = ls.power(frequency)
        period_days = 1 / frequency

        # Use peak_threshold to filter peaks
        peaks, properties = find_peaks(power, height=self.peak_threshold)
        significant_periods = period_days[peaks]
        significant_powers = power[peaks]

        if filter_alias:
            significant_periods, significant_powers = self._filter_aliases(frequency, power, significant_periods, significant_powers)
        

        print("Significant peaks:")
        for period, pwr in zip(significant_periods, significant_powers):
            print(f"Period: {period:.6f} days, Power: {pwr:.6f}")

        
        return frequency, power, period_days, significant_periods, significant_powers

    
    def _filter_aliases(self, frequency, power, periods, powers):

        window_ls = LombScargle(self.data[:, 1], np.ones_like(self.detrended_flux))
        window_power = window_ls.power(frequency)
        
        window_peaks, _ = find_peaks(window_power, height=0.10)
        window_periods = 1 / frequency[window_peaks]

        filtered_periods = []
        filtered_powers = []

        for period, pwr in zip(periods, powers):
            if not any(np.isclose(period, wp, rtol=0.05) for wp in window_periods):
                filtered_periods.append(period)
                filtered_powers.append(pwr)

        return np.array(filtered_periods), np.array(filtered_powers)

    def phasefold(self, period):
        t = self.data[:, 1]
        p = period
        phi = (t % p) / p  
        sorted_indices = np.argsort(phi)
        return phi[sorted_indices], self.detrended_flux[sorted_indices]

    def _identify_harmonics(self, significant_periods, significant_powers, tolerance=0.05):
        harmonics_info = []

        for i, period in enumerate(significant_periods):
            for j, other_period in enumerate(significant_periods):
                if i != j:
                    # Calculate the ratio and check if it's close to an integer (harmonic)
                    ratio = other_period / period
                    harmonic_order = round(ratio)

                    if np.isclose(ratio, harmonic_order, rtol=tolerance):
                        harmonic_period = harmonic_order * period

                        # Check if this harmonic period is close to the other period
                        if np.isclose(harmonic_period, other_period, rtol=tolerance):
                            harmonics_info.append((i + 1, j + 1, harmonic_order, other_period))

        return harmonics_info


    def plot(self, save_dir=None):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        axs[0].plot(self.data[:, 1], self.data[:, 4], color='g', label="Original Light Curve")
        axs[0].set_xlabel("Time (BTJD)")
        axs[0].set_ylabel("Magnitude")
        axs[0].set_title(f"Original Light Curve of {self.target}")
        axs[0].invert_yaxis()
        axs[0].legend()

        axs[1].plot(self.data[:, 1], self.detrended_flux, color='r', label="Detrended Light Curve")
        axs[1].set_xlabel("Time (BTJD)")
        axs[1].set_ylabel("Detrended Magnitude")
        axs[1].set_title(f"Detrended Light Curve of {self.target}")
        axs[1].invert_yaxis()
        axs[1].legend()

        plt.tight_layout()
        fig.set_size_inches(18.5, 10.5)
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"light_curve_{self.target}_{self.x_min}-{self.x_max}.png"))
        else:
            plt.show()

        frequency, power, period_days, significant_periods, significant_powers = self.lomb_scargle()
        fig, ax = plt.subplots(figsize=(18.5, 10.5))

        ax.plot(period_days, power, color='b')
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Power")
        ax.set_title(f"LS Periodogram of {self.target}")
        ax.set_xscale('log')

        text_content = []
        harmonics_info = self._identify_harmonics(significant_periods, significant_powers)

        if significant_periods.size > 0:
            colors = plt.cm.jet(np.linspace(0, 1, len(significant_periods)))  # Generate colors

            for i, (period, pwr) in enumerate(zip(significant_periods, significant_powers)):
                ax.plot(period, pwr, 'o', color=colors[i], markersize=5, label=f'Peak {i + 1}')
                ax.annotate(f'Peak {i + 1}', 
                            (period, pwr),
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center',
                            color=colors[i])
                text_content.append(f"Peak {i + 1}: Period = {period:.6f} days, Power = {pwr:.6f}")

            for fundamental_index, harmonic_index, order, harmonic_period in harmonics_info:
                harmonic_info = f"Harmonic {order} of Peak {fundamental_index} = Peak {harmonic_index} (Period = {harmonic_period:.6f} days)"
                text_content.append(harmonic_info)

            plt.text(0.95, 0.95, "\n".join(text_content), 
                    fontsize=12, va='top', ha='right', 
                    transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.5))

        ax_inset = fig.add_axes([0.6, 0.36, 0.28, 0.3])
        ax_inset.plot(self.data[:, 1], self.detrended_flux, color='r')
        ax_inset.set_xlabel("Time (BTJD)")
        ax_inset.set_ylabel("Detrended Magnitude")
        ax_inset.set_title(f"Detrended Light Curve of {self.target}")
        ax_inset.invert_yaxis()

        if save_dir:
            plt.savefig(os.path.join(save_dir, f"periodogram_{self.target}_{self.x_min}-{self.x_max}.png"))
        else:
            plt.show()

        if self.period is not None:
            plt.figure()
            phi_sorted, flux_sorted = self.phasefold(self.period)
            trend_coeffs = np.polyfit(phi_sorted, flux_sorted, deg=7)
            trend_line = np.polyval(trend_coeffs, phi_sorted)
            plt.scatter(phi_sorted, flux_sorted, color='m', s=5)
            plt.plot(phi_sorted, trend_line, color='b', label="Trend Line", linewidth=2)
            plt.xlabel("Phase")
            plt.ylabel("Magnitude")
            plt.title("Phase Folded Curve")
            plt.gca().invert_yaxis()
            plt.gcf().set_size_inches(18.5, 10.5)

            period_text = f"Period: {self.period:.6f} days"
            plt.text(0.95, 0.95, period_text, 
                    horizontalalignment='right', 
                    verticalalignment='top', 
                    transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.5))

            if self.power is not None:
                power_text = f"Power: {self.power:.6f}"
                plt.text(0.95, 0.90, power_text, 
                        horizontalalignment='right', 
                        verticalalignment='top', 
                        transform=plt.gca().transAxes, 
                        bbox=dict(facecolor='white', alpha=0.5))

            if save_dir:
                plt.savefig(os.path.join(save_dir, f"phase_folded_curve_{self.target}_{self.x_min}-{self.x_max}_{self.period}days.png"))
            else:
                plt.show()


