import numpy as np
import matplotlib.pyplot as plt
from LightCurveObject import LightCurve
from LCWC import LightCurveWithConstraint

filename = r"C:\Users\nic51\Code_Repos\Tess\Data\Interesting_Curves\sector65\2023iwl.txt"

#3090.5-3095

lc = LightCurve(filename)
x_min = 3090.5
x_max = 3095
period = .89
lc_wc = LightCurveWithConstraint(filename, x_min=x_min, x_max=x_max, period=period)

lc.plot()
lc_wc.plot()
save_dir = r"C:\Users\nic51\OneDrive\Desktop\2023iwl"
lc_wc.save_plots(save_dir)
