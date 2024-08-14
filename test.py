import numpy as np
import matplotlib.pyplot as plt
from LightCurveObject import LightCurve
from LCWC import LightCurveWithConstraint

# Step 1: Create synthetic light curve data with known harmonics
np.random.seed(42)  # For reproducibility

t = np.linspace(0, 10, 1000)  # 10 days of data with high cadence
P1 = 2.0  # Primary period in days
P2 = P1 / 2  # Harmonic period in days
flux = 0.5 * np.sin(2 * np.pi * t / P1) + 0.3 * np.sin(2 * np.pi * t / P2)

# Adding some noise to simulate real data
noise = 0.1 * np.random.normal(size=t.shape)
flux += noise

# Save the synthetic data to a file (if needed)
synthetic_filename = "synthetic_light_curve.txt"
np.savetxt(synthetic_filename, np.column_stack([np.zeros_like(t), t, np.zeros_like(t), np.zeros_like(t), flux]))

# Step 2: Use LightCurveWithConstraint to analyze the synthetic data
lc_wc = LightCurveWithConstraint(synthetic_filename, period=P1, target='synthetic', peak_threshold=0.1)
lc_wc.detrend(max_degree=0)  # No detrending needed for synthetic data
lc_wc.plot()

# Step 3: Print and verify the detected harmonics
frequency, power, period_days, significant_periods, significant_powers = lc_wc.lomb_scargle()
harmonics_info = lc_wc._identify_harmonics(significant_periods, significant_powers)

print("Detected Harmonics:")
for fundamental_index, harmonic_index, order, harmonic_period in harmonics_info:
    print(f"Harmonic {order} of Peak {fundamental_index} = Peak {harmonic_index} (Period = {harmonic_period:.6f} days)")

# Expected output should show that Peak 1 is the fundamental with period ~2.0 days
# and Peak 2 is a harmonic with period ~1.0 day (P2 = P1/2).
