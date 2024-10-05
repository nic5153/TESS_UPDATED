import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter
from astropy.io import fits
import tkinter as tk
from matplotlib.colors import Normalize
import gc
from tkinter import filedialog

fits_file = r"C:\Users\nic51\OneDrive\Desktop\2019muu FITS\tess-s0081-2-2_291.874208_33.051450_10x10_astrocut.fits"

def get_total_frames(fits_file):
    with fits.open(fits_file, memmap=True) as hdul:
        pixel_data = hdul['PIXELS'].data
        flux_data = pixel_data['FLUX']
        return flux_data.shape[0]

def load_single_frame(fits_file, frame_number):
    with fits.open(fits_file, memmap=True) as hdul:
        pixel_data = hdul['PIXELS'].data
        flux_data = pixel_data['FLUX']
        return flux_data[frame_number]

def show_slider(total_frames):
    initial_frame = 0
    fig, ax = plt.subplots()

    flux_data_first_frame = load_single_frame(fits_file, initial_frame)

    norm = Normalize(vmin=flux_data_first_frame.min(), vmax=flux_data_first_frame.max())
    
    im = ax.imshow(flux_data_first_frame, cmap='coolwarm', origin='lower', norm=norm)
    plt.colorbar(im, label='Flux (e⁻/s)')
    ax.set_title(f'TESS Image - Time Step {initial_frame}')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')

    ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, 'Time Step', 0, total_frames - 1, valinit=initial_frame, valstep=1)

    def update_slider(val):
        frame = int(val)
        flux_frame = load_single_frame(fits_file, frame)
        im.set_data(flux_frame)
        ax.set_title(f'TESS Image - Time Step {frame}')
        plt.draw()

    slider.on_changed(update_slider)
    plt.tight_layout()
    plt.show()

def show_animation(total_frames, frame_skip=1):
    fig, ax = plt.subplots()

    flux_data_first_frame = load_single_frame(fits_file, 0)


    norm = Normalize(vmin=flux_data_first_frame.min(), vmax=flux_data_first_frame.max())

    im = ax.imshow(flux_data_first_frame, cmap='coolwarm', origin='lower', norm=norm)
    plt.colorbar(im, label='Flux (e⁻/s)')

    def update(frame):
        flux_frame = load_single_frame(fits_file, frame)
        ax.clear()
        im = ax.imshow(flux_frame, cmap='coolwarm', origin='lower', norm=norm)
        ax.set_title(f'TESS Image 2019muu_s81 - Time Step {frame}')
        plt.draw()


    frames_to_display = range(0, total_frames, frame_skip)

    ani = FuncAnimation(fig, update, frames=frames_to_display, repeat=False)

    gif_file = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF files", "*.gif"), ("All files", "*.*")])
    
    if gif_file:
        print("Saving GIF, this may take a while...")
        ani.save(gif_file, writer=PillowWriter(fps=20))
        print(f"GIF saved to: {gif_file}")

    plt.show()

def choose_method(total_frames):
    root = tk.Tk()
    root.title("Choose Visualization Method")

    def on_slider():
        root.destroy()
        show_slider(total_frames)

    def on_animation():
        root.destroy()
        show_animation(total_frames, frame_skip=10)

    tk.Label(root, text="Choose how to visualize the FITS data:").pack(pady=10)
    tk.Button(root, text="Slider", command=on_slider).pack(pady=5)
    tk.Button(root, text="Animation", command=on_animation).pack(pady=5)

    root.mainloop()

total_frames = get_total_frames(fits_file)
choose_method(total_frames)




