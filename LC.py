import numpy as np
import os
import shutil


class SectorProcessor:
  def __init__(self, png_folder, txt_folder, destination_folder):
    self.png_folder = png_folder
    self.txt_folder = txt_folder
    self.destination_folder = destination_folder

