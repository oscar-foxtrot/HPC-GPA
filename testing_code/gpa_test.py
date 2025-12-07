import numpy as np
import matplotlib.pyplot as plt

def CPA(X, Y, scale=True):
    '''
    Procrustes analysis with scaling to align X to Y.
    Both X and Y are arrays of shape (m, n) where m and n are arbitrary.
    Returns: aligned_Y, rotation_matrix, scale, translation
    '''
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X0 = X - X_mean
    Y0 = Y - Y_mean

    U, _, Vt = np.linalg.svd(X0.T @ Y0)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    s = 1
    if scale:
        s = np.trace((X0 @ R).T @ Y0) / np.trace(X0.T @ X0)

    aligned_X = s * X0 @ R + Y_mean

    return aligned_X, R, s, Y_mean - s * X_mean @ R


kpts0 = np.load(f'data/file_469_dynamic.npy', allow_pickle=True)[:]
kpts1 = np.load(f'data/file_474_dynamic.npy', allow_pickle=True)[:]
kpts0 = np.array(kpts0, dtype=np.float64)
kpts1 = np.array(kpts1, dtype=np.float64)

point_set_0 = kpts0[50]
point_set_1 = kpts1[40]


X = np.stack([point_set_0[:, 0], -point_set_0[:, 1]], axis=1)
Y = np.stack([-point_set_1[:, 0], -point_set_1[:, 1]], axis=1)

'''
X = np.array([[0, 0],[1, 1]])
Y = np.array([[0, 0],[-1, 1.25]])
'''

plt.figure(figsize=(6, 6))


plt.scatter(Y[:, 0], Y[:, 1],
            s=50, label="Y that we align to", color='navy')

plt.scatter(X[:, 0], X[:, 1],
            s=35, label="X to be aligned", color='red')

X_aligned = CPA(X, Y, scale=True)[0]

plt.scatter(X_aligned[:, 0], X_aligned[:, 1],
            s=20, label="X aligned to Y", color='lime')


plt.xlabel("X")
plt.ylabel("Y")
plt.title("2D Keypoints (XY projection)")
plt.axis("equal")   # important: preserves geometry
plt.legend()
plt.show()

print(X - Y)
print(X_aligned - Y)
print(np.linalg.norm(X - Y, 'fro'))
print(np.linalg.norm(X_aligned - Y, 'fro'))