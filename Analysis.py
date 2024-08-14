from LightCurveObject import LightCurve
from LCWC import LightCurveWithConstraint
import numpy as np


filename = r"C:\Users\nic51\Code_Repos\Tess\Data\Extracted_Tarballs\TXT_Files\2019muu.txt"
save_dir = r"C:\Users\nic51\OneDrive\Desktop"
target = '2019muu'
x_min = 1689.5
x_max = 1710
period = .182628
power = .444598
peak_threshold = 0.09

lc = LightCurve(filename, target=target)
lc.plot()

lc_wc = LightCurveWithConstraint(filename, x_min=x_min, x_max=x_max, period=period, power=power, target=target, peak_threshold=peak_threshold)
lc_wc.detrend(max_degree=5)
lc_wc.plot()
lc.save_plots(save_dir)
lc_wc.plot(save_dir=save_dir)
