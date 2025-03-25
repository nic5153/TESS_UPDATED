import os
from astropy.io import fits
import numpy as np
from tkinter import filedialog, Tk, Checkbutton, IntVar, Button, Toplevel, Label, Entry

# Convert raw calibration data into master calibration frames
def create_master_frame(frame_dir, frame_type):
    frame_files = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.fits') or f.endswith('.fit')]

    if not frame_files:
        print(f"No {frame_type} frames found in directory: {frame_dir}")
        return None

    stack = []
    for frame_file in frame_files:
        with fits.open(frame_file) as hdul:
            data = hdul[0].data.astype(np.float64)
            stack.append(data)

    master_frame = np.median(stack, axis=0)
    print(f"Created master {frame_type} frame.")
    return master_frame

# Calibrate science frames using master calibration frames
def calibrate_science_frames(science_dir, master_bias, master_dark, master_flat, output_dir):
    master_flat = (master_flat / np.mean(master_flat)) + 1e-6

    science_files = [os.path.join(science_dir, f) for f in os.listdir(science_dir) if f.endswith('.fits') or f.endswith('.fit')]

    if not science_files:
        print(f"No science frames found in directory: {science_dir}")
        return

    for science_file in science_files:
        with fits.open(science_file) as hdul:
            science_data = hdul[0].data.astype(np.float64)

            calibrated_data = (science_data - master_bias - master_dark) / master_flat
            calibrated_data -= np.median(calibrated_data)

            output_file = os.path.join(output_dir, f"calibrated_{os.path.basename(science_file)}")
            fits.writeto(output_file, calibrated_data, hdul[0].header, overwrite=True)
            print(f"Calibrated frame saved to {output_file}")

# Select filters to process
def select_filters():
    """
    Allows the user to select filters to process using a GUI with checkboxes.
    Returns a list of selected filters.
    """
    filter_selection = {}

    def on_submit():
        for filter_name, var in filter_vars.items():
            filter_selection[filter_name] = var.get()
        filter_window.destroy()

    # Create the GUI window
    filter_window = Toplevel()
    filter_window.title("Select Filters")
    filter_vars = {"R": IntVar(), "G": IntVar(), "B": IntVar()}

    for i, (filter_name, var) in enumerate(filter_vars.items()):
        Checkbutton(filter_window, text=f"{filter_name} band", variable=var).grid(row=i, sticky='w')

    Button(filter_window, text="Submit", command=on_submit).grid(row=len(filter_vars), sticky='e')
    filter_window.wait_window()
    return [f for f, selected in filter_selection.items() if selected]

# Select directories for each calibration step
def select_directories(filter_band):
    root = Tk()
    root.withdraw()

    print(f"Select bias frame directory for {filter_band} band:")
    bias_dir = filedialog.askdirectory(title=f"Select Bias Frame Directory for {filter_band} band")

    print(f"Select dark frame directory for {filter_band} band:")
    dark_dir = filedialog.askdirectory(title=f"Select Dark Frame Directory for {filter_band} band")

    print(f"Select flat frame directory for {filter_band} band:")
    flat_dir = filedialog.askdirectory(title=f"Select Flat Frame Directory for {filter_band} band")

    print(f"Select science frame directory for {filter_band} band:")
    science_dir = filedialog.askdirectory(title=f"Select Science Frame Directory for {filter_band} band")

    print(f"Select output directory for calibrated frames for {filter_band} band:")
    output_dir = filedialog.askdirectory(title=f"Select Output Directory for {filter_band} band")

    return bias_dir, dark_dir, flat_dir, science_dir, output_dir

# Main execution
def main():
    # Allow user to select filters
    selected_filters = select_filters()
    if not selected_filters:
        print("No filters selected. Exiting...")
        return

    # Process each filter
    for filter_band in selected_filters:
        print(f"Processing {filter_band} band...")

        bias_dir, dark_dir, flat_dir, science_dir, output_dir = select_directories(filter_band)

        # Create and process calibration frames
        print("Creating master calibration frames...")
        master_bias = create_master_frame(bias_dir, "bias")
        master_dark = create_master_frame(dark_dir, "dark")
        master_flat = create_master_frame(flat_dir, "flat")

        if master_bias is not None and master_dark is not None and master_flat is not None:
            print(f"Applying calibration to science frames for {filter_band} band...")
            calibrate_science_frames(science_dir, master_bias, master_dark, master_flat, output_dir)
        else:
            print(f"Calibration frames are incomplete for {filter_band} band. Skipping...")

if __name__ == "__main__":
    main()

