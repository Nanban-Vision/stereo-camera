import numpy as np

data = np.load('calibration_data.npz')
mtx_left = data['mtx_left']
dist_left = data['dist_left']
mtx_right = data['mtx_right']
dist_right = data['dist_right']
R = data['R']
T = data['T']
E = data['E']
F = data['F']

print(f"data:{data}")
print(f"mtx_left:{mtx_left}")
print(f"dist_left:{dist_left}")
print(f"mtx_right:{mtx_right}")
print(f"dist_right:{dist_right}")
print(f"R:{R}")
print(f"T:{T}")
print(f"E:{E}")
print(f"F:{F}")

