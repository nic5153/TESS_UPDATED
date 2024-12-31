import os
from astropy.io import fits
import numpy as np
from tkinter import filedialog, Tk, Checkbutton, IntVar, Button, Toplevel, Label, Entry
import requests

# Astrometry.net API URL and key
ASTROMETRY_API_URL = "http://nova.astrometry.net/api/"

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

# Submit a plate-solving request to Astrometry.net API
def submit_plate_solve(image_path, api_key):
    print(f"Submitting {image_path} for plate solving...")
    headers = {"User-Agent": "Astrometry Plate Solver"}

    # Step 1: Login
    login_data = {"request": "login", "apikey": api_key}
    response = requests.post(ASTROMETRY_API_URL + "login", data=login_data, headers=headers)
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return False

    # Step 2: Upload image
    with open(image_path, 'rb') as image_file:
        upload_data = {"publicly_visible": "y", "allow_modifications": "y"}
        upload_response = requests.post(ASTROMETRY_API_URL + "upload", files={"file": image_file}, data=upload_data, headers=headers)

    if upload_response.status_code != 200:
        print(f"Upload failed for {image_path}: {upload_response.text}")
        return False

    print(f"Plate solving submitted for {image_path}. Check your Astrometry.net account for results.")
    return True

# Input observatory details
def input_observatory():
    observatory_details = {}

    def on_submit():
        observatory_details['name'] = observatory_name_var.get()
        observatory_details['latitude'] = float(latitude_var.get())
        observatory_details['longitude'] = float(longitude_var.get())
        observatory_details['altitude'] = float(altitude_var.get())
        obs_window.destroy()

    obs_window = Toplevel()
    obs_window.title("Input Observatory Location")

    Label(obs_window, text="Observatory Name:").grid(row=0, column=0, sticky='w')
    observatory_name_var = Entry(obs_window)
    observatory_name_var.insert(0, "Preston Gott Skyview Observatory, Shallowater, Texas")
    observatory_name_var.grid(row=0, column=1, sticky='w')

    Label(obs_window, text="Latitude (degrees):").grid(row=1, column=0, sticky='w')
    latitude_var = Entry(obs_window)
    latitude_var.insert(0, "33.5927")  # Default latitude
    latitude_var.grid(row=1, column=1, sticky='w')

    Label(obs_window, text="Longitude (degrees):").grid(row=2, column=0, sticky='w')
    longitude_var = Entry(obs_window)
    longitude_var.insert(0, "-101.9362")  # Default longitude
    longitude_var.grid(row=2, column=1, sticky='w')

    Label(obs_window, text="Altitude (meters):").grid(row=3, column=0, sticky='w')
    altitude_var = Entry(obs_window)
    altitude_var.insert(0, "972")  # Default altitude
    altitude_var.grid(row=3, column=1, sticky='w')

    Button(obs_window, text="Submit", command=on_submit).grid(row=4, columnspan=2, sticky='e')
    obs_window.wait_window()
    return observatory_details

# Select filters to process
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

# Select directories for each filter
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
selected_filters = select_filters()
observatory_details = input_observatory()
astrometry_api_key = input("Enter your Astrometry.net API key (or press Enter to skip): ").strip()

for filter_band in selected_filters:
    print(f"Processing {filter_band} band...")

    bias_dir, dark_dir, flat_dir, science_dir, output_dir = select_directories(filter_band)

    print("Creating master calibration frames...")
    master_bias = create_master_frame(bias_dir, "bias")
    master_dark = create_master_frame(dark_dir, "dark")
    master_flat = create_master_frame(flat_dir, "flat")

    if master_bias is not None and master_dark is not None and master_flat is not None:
        print(f"Applying calibration to science frames for {filter_band} band...")
        calibrate_science_frames(science_dir, master_bias, master_dark, master_flat, output_dir)

        if astrometry_api_key:
            print(f"Starting plate solving process for {filter_band} band...")
            for science_file in os.listdir(output_dir):
                if science_file.startswith("calibrated_"):
                    submit_plate_solve(os.path.join(output_dir, science_file), astrometry_api_key)
        else:
            print("Skipping plate solving as no API key was provided.")
    else:
        print(f"Calibration frames are incomplete for {filter_band} band. Skipping...")

