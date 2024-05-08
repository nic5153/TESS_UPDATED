import numpy as np
import matplotlib.pyplot as plt
from LightCurveObject import LightCurve

filename = r"C:\Users\nic51\Code_Repos\Tess\Data\Interesting_Curves\sector23\Super Outbursts\2020gbb.txt"

lc = LightCurve(filename)

lc.plot()