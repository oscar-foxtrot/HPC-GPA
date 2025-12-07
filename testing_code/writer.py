import numpy as np

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

for i in range(len(X_list)):
    np.savetxt(f"X_{i}.txt", X_list[i], fmt="%.17g")