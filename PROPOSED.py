import time
import numpy as np

def PROPOSED(X,fitness,lower_bound, upper_bound, max_iterations):
    search_agents,dimension = X.shape[0],X.shape[1]
    fit = np.array([fitness(X[i, :]) for i in range(search_agents)])
    # Best solution initialization
    best_idx = np.argmin(fit)
    X_best = X[best_idx, :].copy()
    f_best = fit[best_idx]

    best_so_far = np.zeros(max_iterations)
    ct = time.time()
    for t in range(1, max_iterations + 1):
        SW = X_best.copy()  # Strongest walrus

        for i in range(search_agents):
            # Phase 1: Feeding Strategy (Exploration)
            I = np.round(1 + np.random.rand())
            X_P1 = X[i, :] + np.random.rand(dimension) * (SW - I * X[i, :])
            X_P1 = np.clip(X_P1, lower_bound, upper_bound)

            F_P1 = fitness(X_P1)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1

            # Phase 2: Migration
            I = -t * ((-1) / max_iterations)  # Updated
            K = np.random.permutation(search_agents)
            K = K[K != i]  # Ensure K is not the same as i
            X_K = X[K[0], :]
            F_RAND = fit[K[0]]

            if fit[i] > F_RAND:
                X_P2 = X[i, :] + np.random.rand() * (X_K - I * X[i, :])
            else:
                X_P2 = X[i, :] + np.random.rand() * (X[i, :] - X_K)

            X_P2 = np.clip(X_P2, lower_bound, upper_bound)

            F_P2 = fitness(X_P2)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2

            # Phase 3: Escaping & Fighting Against Predators (Exploitation)
            LO_LOCAL = lower_bound / t
            HI_LOCAL = upper_bound / t

            X_P3 = X[i, :] + LO_LOCAL + np.random.rand() * (HI_LOCAL - LO_LOCAL)
            X_P3 = np.clip(X_P3, lower_bound, upper_bound)

            F_P3 = fitness(X_P3)
            if F_P3 < fit[i]:
                X[i, :] = X_P3
                fit[i] = F_P3

        # Update best solution
        best_idx = np.argmin(fit)
        if fit[best_idx] < f_best:
            f_best = fit[best_idx]
            X_best = X[best_idx, :].copy()

        best_so_far[t - 1] = f_best
    ct = time.time()-ct
    return f_best,best_so_far, X_best,ct
