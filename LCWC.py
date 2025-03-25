import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from numpy.polynomial import Polynomial
import tkinter as tk
from tkinter import filedialog, messagebox

class LightCurveWithConstraint:
    def __init__(self, filename, x_min=None, x_max=None, cadence_days=0.00277778,
                 period=None, power=None, target=None, peak_threshold=0.09,
                 time_col=0, flux_col=1, data_type="mag"):
        self.filename = filename
        self.x_min = x_min
        self.x_max = x_max
        self.cadence_days = cadence_days
        self.period = period
        self.power = power
        self.target = target
        self.peak_threshold = peak_threshold
        self.data_type = data_type.lower()
        self.time_col = time_col
        self.flux_col = flux_col
        print(f"[DEBUG] Using columns: time={self.time_col}, value={self.flux_col}, data_type={self.data_type}")
        self.data = self._read_data()
        self.detrended_flux = None
        self.num_phases = 1

    def _read_data(self):
        try:
            data = np.loadtxt(self.filename, skiprows=1, usecols=(self.time_col, self.flux_col))
            if self.data_type == "mag":
                data = data[data[:, 1] > 0]
            else:
                data = data[data[:, 1] >= 0]
            time = data[:, 0]
            if self.x_min is not None and self.x_max is not None:
                mask = (time >= self.x_min) & (time <= self.x_max)
                return data[mask]
            elif self.x_min is not None:
                return data[time >= self.x_min]
            elif self.x_max is not None:
                return data[time <= self.x_max]
            else:
                return data
        except Exception as e:
            print(f"Error reading data: {e}")
            data = np.loadtxt(self.filename, skiprows=1, usecols=(self.time_col, self.flux_col))
            if self.data_type == "mag":
                data = data[data[:, 1] > 0]
            else:
                data = data[data[:, 1] >= 0]
            return data

    def _compute_aic(self, val, trend, k):
        resid = val - trend
        sse = np.sum(resid**2)
        aic = 2 * k + len(val) * np.log(sse / len(val))
        return aic

    def detrend(self, max_degree=5):
        val = self.data[:, 1]
        t = self.data[:, self.time_col]
        if max_degree == 0:
            self.detrended_flux = val - np.mean(val)
            return
        best_aic = np.inf
        best_trend = None
        for degree in range(1, max_degree + 1):
            p = Polynomial.fit(t, val, deg=degree)
            trend = p(t)
            aic = self._compute_aic(val, trend, degree + 1)
            if aic < best_aic:
                best_aic = aic
                best_trend = trend
        self.detrended_flux = val - best_trend
        mean_val = np.mean(val)
        self.detrended_flux -= np.mean(self.detrended_flux) - mean_val
        if np.mean(self.detrended_flux) > mean_val:
            self.detrended_flux = mean_val - self.detrended_flux

    def plot_raw(self):
        time = self.data[:, 0]
        val = self.data[:, 1]
        plt.figure(figsize=(10, 6))
        plt.scatter(time, val, color='g', label="Raw Light Curve")
        plt.xlabel("Time (MJD)")
        if self.data_type == "mag":
            plt.ylabel("Magnitude")
            plt.title(f"Raw Light Curve (Magnitude) of {self.target}")
            plt.gca().invert_yaxis()
        elif self.data_type == "flux":
            plt.ylabel("Flux (Jansky)")
            plt.title(f"Raw Light Curve (Flux) of {self.target}")
        plt.legend()
        plt.show()

    def lomb_scargle(self, filter_alias=True):
        if self.detrended_flux is None:
            raise ValueError("Detrended flux not computed. Please run detrend() first.")
        t = self.data[:, 0]
        max_frequency = 0.5 / np.mean(np.diff(t))
        frequency = np.linspace(0.0003, max_frequency, 10000)
        ls = LombScargle(t, self.detrended_flux)
        power = ls.power(frequency)
        period_days = 1 / frequency
        peaks, _ = find_peaks(power, height=self.peak_threshold)
        significant_periods = period_days[peaks]
        significant_powers = power[peaks]
        if filter_alias:
            significant_periods, significant_powers = self._filter_aliases(frequency, power, significant_periods, significant_powers)
        print("Significant peaks:")
        for p_val, pwr in zip(significant_periods, significant_powers):
            print(f"Period: {p_val:.6f} days, Power: {pwr:.6f}")
        return frequency, power, period_days, significant_periods, significant_powers

    def plot_lomb_scargle(self, save_dir=None):
        freq, power, period_days, sig_periods, sig_powers = self.lomb_scargle()
        plt.figure(figsize=(18.5, 10.5))
        plt.plot(period_days, power, 'b-')
        plt.xlabel("Period (days)")
        plt.ylabel("Power")
        plt.title(f"Lomb–Scargle Periodogram for {self.target}")
        plt.xscale('log')
        harmonics_info = self._identify_harmonics(sig_periods, sig_powers)
        text_content = []
        if len(sig_periods) > 0:
            colors = plt.cm.jet(np.linspace(0, 1, len(sig_periods)))
            for i, (p_val, pwr) in enumerate(zip(sig_periods, sig_powers)):
                plt.plot(p_val, pwr, 'o', color=colors[i], markersize=5, label=f'Peak {i+1}')
                plt.annotate(f'Peak {i+1}', (p_val, pwr), textcoords="offset points", xytext=(0,10),
                             ha='center', color=colors[i])
                text_content.append(f"Peak {i+1}: Period = {p_val:.6f} days, Power = {pwr:.6f}")
            for fundamental_index, harmonic_index, order, harmonic_period in harmonics_info:
                harmonic_info = f"Harmonic {order} of Peak {fundamental_index} = Peak {harmonic_index} (Period = {harmonic_period:.6f} days)"
                text_content.append(harmonic_info)
            plt.text(0.95, 0.95, "\n".join(text_content), fontsize=12, va='top', ha='right', 
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"lomb_scargle_{self.target}.png"))
        else:
            plt.show()
        plt.close()

    def _filter_aliases(self, frequency, power, periods, powers):
        window_ls = LombScargle(self.data[:, 0], np.ones_like(self.detrended_flux))
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

    def _identify_harmonics(self, significant_periods, significant_powers, tolerance=0.05):
        harmonics_info = []
        for i, period in enumerate(significant_periods):
            for j, other_period in enumerate(significant_periods):
                if i != j:
                    ratio = other_period / period
                    harmonic_order = round(ratio)
                    if np.isclose(ratio, harmonic_order, rtol=tolerance):
                        harmonic_period = harmonic_order * period
                        if np.isclose(harmonic_period, other_period, rtol=tolerance):
                            harmonics_info.append((i + 1, j + 1, harmonic_order, other_period))
        return harmonics_info

    def phasefold(self, period, num_phases=1):
        t = self.data[:, 0]
        p = period
        phi = (t % p) / p
        sorted_indices = np.argsort(phi)
        phi_sorted = phi[sorted_indices]
        val_sorted = self.detrended_flux[sorted_indices]
        if num_phases > 1:
            phi_multi = np.concatenate([phi_sorted + k for k in range(num_phases)])
            val_multi = np.tile(val_sorted, num_phases)
            return phi_multi, val_multi
        else:
            return phi_sorted, val_sorted

    def plot_phase_fold(self, save_dir=None):
        if self.period is None:
            print("No period provided; cannot phase-fold.")
            return
        plt.figure()
        phi_sorted, val_sorted = self.phasefold(self.period, num_phases=self.num_phases)
        plt.scatter(phi_sorted, val_sorted, color='m', s=20, label="Phase Data")
        plt.xlabel("Phase")
        if self.data_type == "mag":
            plt.ylabel("Magnitude")
        else:
            plt.ylabel("Flux (Jansky)")
        plt.title("Phase Folded Curve")
        if self.data_type == "mag":
            plt.gca().invert_yaxis()
        plt.gcf().set_size_inches(18.5, 10.5)
        period_text = f"Period: {self.period:.6f} days"
        plt.text(0.95, 0.95, period_text,
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.5))
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"phase_folded_curve_{self.target}_{self.x_min}-{self.x_max}_{self.period}days.png"))
        else:
            plt.show()
        plt.close()

    def plot(self, save_dir=None):
        fig, axs = plt.subplots(2, 1, figsize=(18.5, 10.5))
        axs[0].scatter(self.data[:, 0], self.data[:, 1], color='g', label="Original Light Curve")
        axs[0].set_xlabel("Time (MJD)")
        if self.data_type == "mag":
            axs[0].set_ylabel("Magnitude")
            axs[0].set_title(f"Original Light Curve (Magnitude) of {self.target}")
            axs[0].invert_yaxis()
        else:
            axs[0].set_ylabel("Flux (Jansky)")
            axs[0].set_title(f"Original Light Curve (Flux) of {self.target}")
        axs[0].legend()
        axs[1].scatter(self.data[:, 0], self.detrended_flux, color='r', label="Detrended Light Curve")
        axs[1].set_xlabel("Time (MJD)")
        if self.data_type == "mag":
            axs[1].set_ylabel("Detrended Magnitude")
            axs[1].set_title(f"Detrended Light Curve (Magnitude) of {self.target}")
            axs[1].invert_yaxis()
        else:
            axs[1].set_ylabel("Detrended Flux (Jansky)")
            axs[1].set_title(f"Detrended Light Curve (Flux) of {self.target}")
        axs[1].legend()
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"light_curve_{self.target}_detrended.png"))
        else:
            plt.show()
        plt.close()
        self.plot_phase_fold(save_dir=save_dir)

def launch_detrending_ui(lc_wc):
    top = tk.Toplevel()
    top.title("Phase 2A: Detrending Options")
    tk.Label(top, text="X Min:").pack()
    x_min_entry = tk.Entry(top, width=10)
    x_min_entry.pack()
    tk.Label(top, text="X Max:").pack()
    x_max_entry = tk.Entry(top, width=10)
    x_max_entry.pack()
    tk.Label(top, text="Detrending Polynomial Degree:").pack()
    detrend_degree_entry = tk.Entry(top, width=10)
    detrend_degree_entry.pack()
    tk.Label(top, text="Peak Threshold:").pack()
    peak_threshold_entry = tk.Entry(top, width=10)
    peak_threshold_entry.pack()
    def run_detrending():
        x_min = float(x_min_entry.get()) if x_min_entry.get() else None
        x_max = float(x_max_entry.get()) if x_max_entry.get() else None
        detrend_degree = int(detrend_degree_entry.get()) if detrend_degree_entry.get() else 5
        pt = float(peak_threshold_entry.get()) if peak_threshold_entry.get() else lc_wc.peak_threshold
        lc_wc.x_min = x_min
        lc_wc.x_max = x_max
        lc_wc.peak_threshold = pt
        lc_wc.data = lc_wc._read_data()
        lc_wc.detrend(max_degree=detrend_degree)
        lc_wc.plot_lomb_scargle(save_dir=None)
        launch_refinement_ui(lc_wc)
        top.destroy()
    tk.Button(top, text="Run Detrending and Show Lomb–Scargle", command=run_detrending).pack()

def launch_refinement_ui(lc_wc):
    top = tk.Toplevel()
    top.title("Phase 2B: Refinement Options")
    tk.Label(top, text="Period:").pack()
    period_entry = tk.Entry(top, width=10)
    period_entry.pack()
    tk.Label(top, text="Power:").pack()
    power_entry = tk.Entry(top, width=10)
    power_entry.pack()
    tk.Label(top, text="Number of Phases (1-4):").pack()
    num_phases_entry = tk.Entry(top, width=10)
    num_phases_entry.pack()
    def run_refinement():
        period = float(period_entry.get()) if period_entry.get() else None
        power_val = float(power_entry.get()) if power_entry.get() else None
        num_phases = int(num_phases_entry.get()) if num_phases_entry.get() else 1
        if num_phases not in [1, 2, 3, 4]:
            num_phases = 1
        lc_wc.period = period
        lc_wc.power = power_val
        lc_wc.num_phases = num_phases
        lc_wc.plot_phase_fold(save_dir=None)
        top.destroy()
    tk.Button(top, text="Run Additional Analysis", command=run_refinement).pack()

def launch_ui():
    root = tk.Tk()
    root.title("Phase 1: Select Light Curve")
    tk.Label(root, text="Select Light Curve Directory:").pack()
    file_dir_entry = tk.Entry(root, width=50)
    file_dir_entry.pack()
    def select_directory_for_files():
        directory = filedialog.askdirectory()
        file_dir_entry.delete(0, tk.END)
        file_dir_entry.insert(0, directory)
    tk.Button(root, text="Browse", command=select_directory_for_files).pack()
    tk.Label(root, text="Enter Light Curve File Name/Pattern:").pack()
    file_pattern_entry = tk.Entry(root, width=30)
    file_pattern_entry.pack()
    def update_file_list():
        directory = file_dir_entry.get()
        pattern = file_pattern_entry.get()
        file_listbox.delete(0, tk.END)
        if directory:
            try:
                files = os.listdir(directory)
                matching_files = [f for f in files if (pattern in f if pattern else True)
                                  and os.path.isfile(os.path.join(directory, f))]
                for f in matching_files:
                    file_listbox.insert(tk.END, f)
            except Exception as e:
                print("Error listing files:", e)
    tk.Button(root, text="Refresh File List", command=update_file_list).pack()
    file_listbox = tk.Listbox(root, width=50, height=10)
    file_listbox.pack()
    tk.Label(root, text="Selected File:").pack()
    file_chosen_entry = tk.Entry(root, width=50)
    file_chosen_entry.pack()
    def on_file_select(event):
        selection = file_listbox.curselection()
        if selection:
            index = selection[0]
            selected = file_listbox.get(index)
            full_path = os.path.join(file_dir_entry.get(), selected)
            file_chosen_entry.delete(0, tk.END)
            file_chosen_entry.insert(0, full_path)
    file_listbox.bind('<<ListboxSelect>>', on_file_select)
    tk.Label(root, text="Time Column Index (default 0):").pack()
    time_col_entry = tk.Entry(root, width=10)
    time_col_entry.pack()
    tk.Label(root, text="Value Column Index (default 1):").pack()
    flux_col_entry = tk.Entry(root, width=10)
    flux_col_entry.pack()
    tk.Label(root, text="Select Data Type:").pack()
    data_type_var = tk.StringVar(value="mag")
    tk.Radiobutton(root, text="Magnitude", variable=data_type_var, value="mag").pack()
    tk.Radiobutton(root, text="Flux (Jansky)", variable=data_type_var, value="flux").pack()
    def plot_light_curve():
        filename = file_chosen_entry.get()
        if not filename:
            print("Please select a file.")
            return
        time_col = int(time_col_entry.get()) if time_col_entry.get() else 0
        flux_col = int(flux_col_entry.get()) if flux_col_entry.get() else 1
        data_type = data_type_var.get().strip().lower()
        print(f"[DEBUG] Plotting with columns: time={time_col}, value={flux_col}, data_type={data_type}")
        lc_wc = LightCurveWithConstraint(filename=filename,
                                         target="Raw Light Curve",
                                         peak_threshold=0.09,
                                         time_col=time_col,
                                         flux_col=flux_col,
                                         data_type=data_type)
        lc_wc.plot_raw()
        launch_detrending_ui(lc_wc)
    tk.Button(root, text="Plot Light Curve", command=plot_light_curve).pack()
    root.mainloop()

if __name__ == "__main__":
    launch_ui()
