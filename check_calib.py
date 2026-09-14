import numpy as np
data = np.load("AGZ_subset/calibration_data.npz")
print("Calibration Keys:", data.files)
for key in data.files:
    print(f"\n--- {key} ---")
    print(data[key])