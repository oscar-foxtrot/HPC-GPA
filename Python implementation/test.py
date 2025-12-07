import numpy as np
import matplotlib.pyplot as plt

from gpa import CPA, GPA

kpts0 = np.load(f'data/file_469_dynamic.npy', allow_pickle=True)[:]
kpts1 = np.load(f'data/file_474_dynamic.npy', allow_pickle=True)[:]
kpts0 = np.array(kpts0, dtype=np.float64)
kpts1 = np.array(kpts1, dtype=np.float64)

point_set_0 = kpts0[50]
point_set_1 = kpts1[40]
point_set_2 = kpts1[35]

X = 0.25 * np.stack([point_set_0[:, 0], -point_set_0[:, 1]], axis=1)
Y = 0.25 * np.stack([-point_set_1[:, 0], -point_set_1[:, 1]], axis=1) @ np.array([[-1, 0],[0, -1]])
Z = 0.25 * np.stack([-point_set_2[:, 0], -point_set_2[:, 1]], axis=1)

X_list = [X, Y, Z]

point_set_0 = kpts0[70]
point_set_1 = kpts1[60]
point_set_2 = kpts1[55]

X = 0.5 * np.stack([point_set_0[:, 0], -point_set_0[:, 1]], axis=1)
Y = 0.25 * np.stack([-point_set_1[:, 0], -point_set_1[:, 1]], axis=1) @ np.array([[-1, 0],[0, -1]])
Z = 0.75 * np.stack([-point_set_2[:, 0], -point_set_2[:, 1]], axis=1) @ np.array([[-1, 0],[0, -1]])

X_list += [X, Y, Z]

plt.figure(figsize=(6, 6))

for i in range(len(X_list)):
    plt.scatter(X_list[i][:, 0], X_list[i][:, 1],
                s=80 - 5 * i, label=f"X_{i}")


res_gpa = GPA(X_list, scale=True, epsilon=1e-15)

print(res_gpa[2])

for i in range(len(res_gpa[1])):
    plt.scatter(res_gpa[1][i][:, 0], res_gpa[1][i][:, 1],
                s=50 - 5 * i, label=f"X_{i}_aligned")

plt.scatter(res_gpa[0][:, 0], res_gpa[0][:, 1],
                s=20, label=f"Y_consensus")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("2D Keypoints (XY projection)")
plt.axis("equal")
plt.legend()
plt.show()

i = 5
point = 12

cpa_scales = []

for i in range(len(X_list)):
    cpa_scales += [CPA(X_list[i], res_gpa[1][i], scale=True, translation=True)[2]]

print(np.array(cpa_scales))