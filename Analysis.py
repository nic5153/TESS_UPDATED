import numpy as np
import matplotlib.pyplot as plt
from LightCurveObject import LightCurve
from LCWC import LightCurveWithConstraint

filename = r"C:\Users\nic51\Code_Repos\Tess\Data\Interesting_Curves\sector65\2023iwl.txt"

lc = LightCurve(filename)
x_min =3078
x_max =3079
lc_wc = LightCurveWithConstraint(filename, x_min=x_min, x_max=x_max)

lc.plot()
lc_wc.plot()