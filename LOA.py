import numpy as np
import time


def LOA(x, fobj, Lb, Ub, Max_iter):
    beta = 0.8
    u = 0.5
    N, dim = x.shape[0], x.shape[1]
    ub = Ub[0, :]
    lb = Lb[0, :]

    f = fobj(x)

    fgbest = min(f)
    fgworst = max(f)
    igbest = np.where(min(f) == fgbest)
    igworst = np.where(min(f) == fgworst)
    gbest = x[igbest, :]
    gworst = x[igworst, :]
    fpbest = f
    pbest = x

    l = 0
    fbst = np.zeros((Max_iter, 1))
    ct = time.time()
    n = np.zeros((N))
    while l < Max_iter:

        print(l)
        for i in range(N):

            C = (fpbest * (x[i, :])) / fpbest * (x[i, :])

            p = np.exp(-beta * ((fpbest * (x[i, :])) / gworst))

            r = np.random.rand()

            if r > 0.5:

                x[i, :] = x[i, :] + np.random.rand() * (x[i, :] - x[i, :]) + np.random.rand() * (
                        x[i, :] - x[i, :]) + np.random.rand() * C * p * x[i, :]

            else:

                n[i] = np.round(len(N) * u)

                x[i, :] = np.round(x[i, :] - np.random.rand() * x[i, :] * (dim / np.max(dim)))

                x[i, :] = np.round(x[i, :] - np.random.rand() * x[i, :] * (i / N))

            # Calculate objective function
            f = fobj(x[i, :])

            # Could he finds food here
            minf = min(f)
            iminf = np.where(min(f) == minf)
            if minf <= fgbest:
                fgbest = minf
                gbest = x[iminf, :]
                best_sub = x[iminf, :]
                fbst[l] = minf
            else:
                fbst[l] = fgbest
                best_sub = gbest
        inewpb = np.where(f <= fpbest)
        pbest[inewpb, :] = x[inewpb[0], :]
        fpbest[inewpb] = f[inewpb]

    ct = time.time() - ct
    best_fit = fbst[Max_iter - 1]
    best_sub = np.reshape(best_sub, (-1))
    return best_fit, fbst, best_sub, ct
