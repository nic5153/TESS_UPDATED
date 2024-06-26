from LightCurveObject import LightCurve
from LCWC import LightCurveWithConstraint


filename = r"C:\Users\nic51\Code_Repos\Tess\Data\Extracted_Tarballs\TXT_Files\2019pco.txt"
save_dir = r"C:\Users\nic51\OneDrive\Desktop\Super Outbursts\2019pco"
x_min = 1725
x_max = 1731
period = .068711


lc = LightCurve(filename)
lc.plot()

lc_wc = LightCurveWithConstraint(filename, x_min=x_min, x_max=x_max, period=period)
lc_wc.detrend(max_degree=5)
lc_wc.plot()
lc.save_plots(save_dir)
lc_wc.save_plots(save_dir)

