from LightCurveObject import LightCurve
from LCWC import LightCurveWithConstraint
import numpy as np


filename = r"C:\Users\nic51\Code_Repos\Tess\Data\Extracted_Tarballs\TXT_Files\2020qit.txt"
save_dir = r"C:\Users\nic51\OneDrive\Desktop\2020qit"
target = '2020qit'
x_min = 2053.6
x_max = 2058
period = .078245
power = .410110


lc = LightCurve(filename, target=target)
lc.plot()

lc_wc = LightCurveWithConstraint(filename, x_min=x_min, x_max=x_max, period=period, power=power, target=target)
lc_wc.detrend(max_degree=4)
lc_wc.plot()
lc.save_plots(save_dir)
lc_wc.save_plots(save_dir)
