import numpy as np
import matplotlib.pyplot as plt
from LightCurveObject import LightCurve
from LCWC import LightCurveWithConstraint

filename = r"C:\Users\nic51\Code_Repos\Tess\Data\Interesting_Curves\sector65\2023iwl.txt"

#3084-3085.5

lc = LightCurve(filename)
x_min = 3084
x_max = 3085.5
period = 1.81
lc_wc = LightCurveWithConstraint(filename, x_min=x_min, x_max=x_max, period=period)

lc.plot()
lc_wc.plot()
save_dir = r"C:\Users\nic51\OneDrive\Desktop\2023iwl"
lc_wc.save_plots(save_dir)
