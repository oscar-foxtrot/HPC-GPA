import numpy as np
import matplotlib.pyplot as plt

def CPA(X, Y, scale=True, translation=True):
    '''
    Procrustes analysis with optional scaling and translation to align X to Y.
    Both X and Y are arrays of shape (m, n) where m and n are arbitrary.
    Returns: aligned_X, rotation_matrix, scale, translation
    '''

    if translation:
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        X0 = X - X_mean
        Y0 = Y - Y_mean
    else:
        X_mean = np.zeros(X.shape[1], dtype=X.dtype)
        Y_mean = np.zeros(Y.shape[1], dtype=Y.dtype)
        X0 = X
        Y0 = Y

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


def GPA(X_list, scale=True, epsilon=1e-4):
    '''
    Generalized Procrustes analysis with optional scaling to align X to Y.
    Both X and Y are arrays of shape (m, n) where m and n are arbitrary.
    Returns: Y_consensus, aligned_X_list, scaling_factors
    '''

    m = len(X_list)

    # STEP 1
    X_list = [np.array(X.copy()) for X in X_list]
    for i in range(m):
        X_mean = X_list[i].mean(axis=0)
        X_list[i] = X_list[i] - X_mean

    fro_sum = 0
    for i in range(m):
        fro_sum += np.linalg.norm(X_list[i], 'fro')**2

    var_lambda = np.sqrt(m / fro_sum)

    for i in range(m):
        X_list[i] = X_list[i] * var_lambda

    Y_consensus = X_list[0]

    # STEP 2
    for i in range(1, m):
        X_list[i] = CPA(X_list[i], Y_consensus, scale=False, translation=False)[0]

    Y_consensus = np.sum(np.stack(X_list, axis=0), axis=0) / m
    Sr = m * (1 - np.trace(Y_consensus @ Y_consensus.T))
    print(Sr)
    scaling_factors = np.array([var_lambda for _ in range(m)])

    # STEP 3
    while True:
        for i in range(m):
            X_list[i] = CPA(X_list[i], Y_consensus, scale=False, translation=False)[0]

        # STEP 4 (nothing is done in the optimized version)
        # STEP 5
        if scale:
            for i in range(m):
                change_factor = np.sqrt(np.trace(X_list[i] @ Y_consensus.T) / \
                    (np.trace(X_list[i] @ X_list[i].T) * np.trace(Y_consensus @ Y_consensus.T)))
                s_i_star = scaling_factors[i] * change_factor
                X_list[i] *= s_i_star / scaling_factors[i]
                scaling_factors[i] = s_i_star

        Y_consensus_star_star = np.sum(np.stack(X_list, axis=0), axis=0) / m
        Sr_star_star = Sr - m * np.trace(Y_consensus_star_star @ Y_consensus_star_star.T - Y_consensus @ Y_consensus.T)
        print(Sr_star_star)
        Y_consensus = Y_consensus_star_star

        # STEP 6
        if Sr - Sr_star_star < epsilon:
            break
        else:
            Sr = Sr_star_star
    
    # STEP 7
    return Y_consensus, X_list, scaling_factors