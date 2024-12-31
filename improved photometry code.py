import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.time import Time
import pandas as pd
from tkinter import filedialog, Tk, Checkbutton, IntVar, Button, Toplevel, Label, Entry

# Moffat 2D model
def moffat_2d(x, y, x0, y0, amp, alpha, beta, offset):
    r2 = (x - x0)**2 + (y - y0)**2
    return offset + amp * (1 + (r2 / alpha**2))**(-beta)

# Fit a Moffat 2D profile to a single target
def fit_moffat_psf(data, x_guess, y_guess):
    def moffat_fit(xy, x0, y0, amp, alpha, beta, offset):
        return moffat_2d(xy[0], xy[1], x0, y0, amp, alpha, beta, offset)

    y, x = np.mgrid[:data.shape[0], :data.shape[1]]
    p0 = [x_guess, y_guess, np.max(data), 2, 2.5, np.median(data)]  # Initial guesses

    try:
        xy = np.vstack((x.ravel(), y.ravel()))
        popt, _ = curve_fit(moffat_fit, xy, data.ravel(), p0=p0)
        return popt
    except ValueError as e:
        print(f"ValueError during Moffat PSF fitting: {e}")
        return None
    except RuntimeError as e:
        print(f"RuntimeError during Moffat PSF fitting: {e}")
        return None

# Convert RA/DEC (in HMS/DMS) to pixel coordinates
def get_pixel_coords(fits_header, ra_hms, dec_dms):
    wcs = WCS(fits_header)
    ra = sum(float(x) / 60**i for i, x in enumerate(ra_hms.split(':'))) * 15
    dec_sign = 1 if dec_dms[0] != '-' else -1
    dec_parts = dec_dms.lstrip('-').split(':')
    dec = dec_sign * sum(float(x) / 60**i for i, x in enumerate(dec_parts))
    return wcs.world_to_pixel_values(ra, dec)

# Filter selection window
def select_filters():
    filter_selection = {}
    def on_submit():
        for filter_name, var in filter_vars.items():
            filter_selection[filter_name] = var.get()
        filter_window.destroy()

    filter_window = Toplevel()
    filter_window.title("Select Filters")
    filter_vars = {"R": IntVar(), "G": IntVar(), "B": IntVar()}

    for i, (filter_name, var) in enumerate(filter_vars.items()):
        Checkbutton(filter_window, text=f"{filter_name} band", variable=var).grid(row=i, sticky='w')

    Button(filter_window, text="Submit", command=on_submit).grid(row=len(filter_vars), sticky='e')
    filter_window.wait_window()
    return [f for f, selected in filter_selection.items() if selected]

# Target selection window
def select_targets():
    target_selection = {}

    def on_submit():
        target_selection['num_targets'] = num_targets_var.get()
        target_window.destroy()

    target_window = Toplevel()
    target_window.title("Select Number of Targets")
    num_targets_var = IntVar(value=2)  # Default is 2 targets

    Label(target_window, text="Number of Targets:").grid(row=0, column=0, sticky='w')
    Entry(target_window, textvariable=num_targets_var).grid(row=0, column=1, sticky='w')

    Button(target_window, text="Submit", command=on_submit).grid(row=1, columnspan=2, sticky='e')
    target_window.wait_window()
    return target_selection['num_targets']

# RA/DEC input window
def input_targets(num_targets):
    targets = []

    def on_submit():
        for i in range(num_targets):
            ra = ra_vars[i].get()
            dec = dec_vars[i].get()
            targets.append((ra, dec))
        input_window.destroy()

    input_window = Toplevel()
    input_window.title("Input RA and DEC for Targets")

    ra_vars = [Entry(input_window) for _ in range(num_targets)]
    dec_vars = [Entry(input_window) for _ in range(num_targets)]

    for i in range(num_targets):
        Label(input_window, text=f"Target {i+1} RA:").grid(row=i, column=0, sticky='w')
        ra_vars[i].grid(row=i, column=1, sticky='w')

        Label(input_window, text=f"Target {i+1} DEC:").grid(row=i, column=2, sticky='w')
        dec_vars[i].grid(row=i, column=3, sticky='w')

    Button(input_window, text="Submit", command=on_submit).grid(row=num_targets, columnspan=4, sticky='e')
    input_window.wait_window()
    return targets

# Batch process FITS files
def process_fits_with_time(fits_dirs, targets, output_dir):
    results = {}

    for filter_band, fits_dir in fits_dirs.items():
        for target_index, (ra_target, dec_target) in enumerate(targets):
            results[f"Target_{target_index+1}_{filter_band}"] = []

        for fits_file in sorted(os.listdir(fits_dir)):
            if fits_file.endswith(".fits") or fits_file.endswith(".fit"):
                with fits.open(os.path.join(fits_dir, fits_file)) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header

                    # Extract time
                    if 'DATE-OBS' in header:
                        time_obs = Time(header['DATE-OBS']).jd  # Julian Date
                    else:
                        print(f"DATE-OBS not found in {fits_file}, skipping.")
                        continue

                    # Process each target
                    for target_index, (ra_target, dec_target) in enumerate(targets):
                        x, y = get_pixel_coords(header, ra_target, dec_target)

                        # Define a cutout box around the target
                        box_size = 50
                        x_min = int(x - box_size // 2)
                        x_max = int(x + box_size // 2)
                        y_min = int(y - box_size // 2)
                        y_max = int(y + box_size // 2)
                        cutout = data[y_min:y_max, x_min:x_max]

                        # Ensure cutout is valid
                        if cutout.size == 0 or cutout.shape[0] < 3 or cutout.shape[1] < 3:
                            print(f"Invalid cutout for {fits_file}. Skipping.")
                            continue

                        # Fit PSF for target
                        popt = fit_moffat_psf(cutout, x - x_min, y - y_min)
                        if popt is not None:
                            flux = popt[2]

                            # Debugging: Print flux for each target
                            print(f"Target {target_index+1} ({filter_band} band): Flux = {flux}")

                            # Zero-point calculation
                            reference_flux, reference_magnitude = 14096.23, 11.887
                            zero_point = reference_magnitude + 2.5 * np.log10(reference_flux)

                            # Debugging: Print zero-point
                            print(f"Zero-point = {zero_point}")

                            magnitude = zero_point - 2.5 * np.log10(flux) if flux > 0 else np.nan

                            # Debugging: Print calculated magnitude
                            print(f"Target {target_index+1} ({filter_band} band): Magnitude = {magnitude}")

                            results[f"Target_{target_index+1}_{filter_band}"].append({'Time': time_obs, 'Magnitude': magnitude})

                            plt.figure(figsize=(8, 4))
                            plt.subplot(1, 2, 1)
                            plt.imshow(cutout, cmap='gray', origin='lower', vmin=np.percentile(cutout, 5), vmax=np.percentile(cutout, 95))
                            plt.colorbar(label='Pixel Value')
                            plt.title(f'Cutout ({filter_band}) Target {target_index+1}')
                            plt.scatter(x - x_min, y - y_min, color='blue', label=f'Target {target_index+1}')
                            plt.legend()

                            plt.subplot(1, 2, 2)
                            model = moffat_2d(*np.meshgrid(np.arange(cutout.shape[1]), np.arange(cutout.shape[0])), *popt)
                            plt.imshow(model.reshape(cutout.shape), cmap='gray', origin='lower', vmin=np.percentile(model, 5), vmax=np.percentile(model, 95))
                            plt.colorbar(label='Model Value')
                            plt.title(f'Moffat PSF Fit Target {target_index+1}')
                            plt.scatter(x - x_min, y - y_min, color='blue', label=f'Target {target_index+1}')
                            plt.legend()

                            plt.tight_layout()
                            if not hasattr(process_fits_with_time, 'windows_shown'):
                                process_fits_with_time.windows_shown = 0

                            if process_fits_with_time.windows_shown < 3:
                                plt.show()
                                process_fits_with_time.windows_shown += 1
                            else:
                                plt.close()

    for key, data in results.items():
        if data:
            df = pd.DataFrame(data)
            output_file = os.path.join(output_dir, f"{key}.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved {key} data to {output_file}")

# Directory selection
def select_directories():
    root = Tk()
    root.withdraw()

    selected_filters = select_filters()
    fits_dirs = {}
    for filter_band in selected_filters:
        print(f"Select {filter_band} band FITS directory:")
        fits_dirs[filter_band] = filedialog.askdirectory(title=f"Select {filter_band} band FITS directory")

    print("Select output directory for CSV files:")
    output_dir = filedialog.askdirectory(title="Select output directory")

    return fits_dirs, output_dir

fits_dirs, output_dir = select_directories()
num_targets = select_targets()
targets = input_targets(num_targets)
process_fits_with_time(fits_dirs, targets, output_dir)
