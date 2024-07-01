from LightCurveObject import LightCurve
from LCWC import LightCurveWithConstraint
import numpy as np


filename = r"C:\Users\nic51\Code_Repos\Tess\Data\Extracted_Tarballs\TXT_Files\2019muu.txt"
save_dir = r"C:\Users\nic51\OneDrive\Desktop\Super Outbursts\2019muu"
target = '2019muu'
x_min = 1689.5
x_max = 1710
period = .200988
power = 0.322009


lc = LightCurve(filename, target=target)
lc.plot()

lc_wc = LightCurveWithConstraint(filename, x_min=x_min, x_max=x_max, period=period, power=power, target=target)
lc_wc.detrend(max_degree=4)
lc_wc.plot()
lc.save_plots(save_dir)
lc_wc.save_plots(save_dir)
