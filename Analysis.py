import os
from LightCurveObject import LightCurve
from LCWC import LightCurveWithConstraint

filename = r"C:\Users\nic51\Code_Repos\Tess\Data\Interesting_Curves\sector65\2023iwl.txt"
save_dir = r"C:\Users\nic51\OneDrive\Desktop\2023iwl"

x_min = 3084
x_max = 3085
period = .07534

lc = LightCurve(filename)
lc.plot()


lc_wc = LightCurveWithConstraint(filename, x_min=x_min, x_max=x_max, period=period)
lc_wc.detrend(max_degree=6)
lc_wc.plot()


