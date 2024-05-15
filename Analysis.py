import numpy as np
import matplotlib.pyplot as plt
from LightCurveObject import LightCurve
from LCWC import LightCurveWithConstraint

filename = r"C:\Users\nic51\Code_Repos\Tess\Data\Interesting_Curves\sector07\Super Outbursts\2019ait.txt"

lc = LightCurve(filename)
x_min =1514.75
x_max =1515
lc_wc = LightCurveWithConstraint(filename, x_min=x_min, x_max=x_max)

lc.plot()
lc_wc.plot()