import time
from math import inf
import numpy as np

def RouletteWheelSelection(P):
    r = np.random.rand()
    C = np.sum(P)
    i = np.where(r <= C)
    return r

def DMO(Position,F_obj, VarMin, VarMax,MaxIt):
    nPop,nVar = Position.shape[0],Position.shape[1]
    VarSize = np.array([1, nVar])
    nBabysitter = 3
    nAlphaGroup = nPop - nBabysitter
    nScout = nAlphaGroup
    L = np.round(0.6 * nVar * nBabysitter)
    peep = 2
    #######################################################################
    Cost = np.zeros((nPop))
    tau = inf
    Iter = 1
    sm = inf
    # Create Initial Population
    BEF = np.zeros((nPop))
    for i in range(nPop):
        Cost[i] = F_obj(Position[i,:])
    C = np.zeros((nAlphaGroup, 1))
    CF = (1 - Iter / MaxIt) ** (2 * Iter / MaxIt)
    # Array to Hold Best Cost Values
    BestCost = np.zeros((MaxIt, 1))
    ## DMOA Main Loop
    ct = time.time()
    for it in range(MaxIt):
        # Alpha group
        F = np.zeros((nAlphaGroup, 1))
        MeanCost = np.mean(np.array([Cost]))
        for i in range(nAlphaGroup):
            # Calculate Fitness Values and Selection of Alpha
            F[i] = np.exp(-Cost[i] / MeanCost)
        P = F / sum(F)
        # Foraging led by Alpha female
        for m in range(nAlphaGroup):
            # Select Alpha female
            i = RouletteWheelSelection(P)
            # Choose k randomly, not equal to Alpha
            K = np.arange(i - 1 + 1), np.arange(nAlphaGroup + 1)
            k = K[1][np.random.randint(K[1].shape)]
            # Define Vocalization Coeff.
            phi = (peep / 2) * np.random.uniform(- 1, + 1, VarSize)
            # New Mongoose Position
            newPosition = Position[m] + np.multiply(phi, (Position[m] - Position[k]))
            # Evaluation
            newCost = F_obj(newPosition)
            # Comparision
            if newCost <= Cost:
                Position[i] = newPosition
            else:
                C[i] = C[i] + 1
        # Scout group
        for i in range(nScout):
            # Choose k randomly, not equal to i
            K = np.array([np.arange(1, i - 1 + 1), np.arange(i + 1, nAlphaGroup + 1)])
            k = K[np.random.uniform(np.array([1, np.asarray(K).size]))]
            # Define Vocalization Coeff.
            phi = (peep / 2) * np.random.uniform(- 1, + 1, VarSize)
            # New Mongoose Position
            newPosition =Position[i] + np.multiply(phi, (Position[i] - Position[k]))
            # Evaluation
            newCost = F_obj(newPosition)
            # Sleeping mould
            sm[i] = (newCost - Cost[i]) / np.amax(newCost, Cost[i])
            # Comparision
            if newCost <= Cost[i]:
                Position[i] = newPosition
            else:
                C[i] = C[i] + 1
        # Babysitters
        for i in range(nBabysitter):
            if C[i] >= L:
                Position = np.random.uniform(VarMin, VarMax, VarSize)
                Cost = F_obj(Position[i])
                C[i] = 0
        # Update Best Solution Ever Found
        for i in range(nAlphaGroup):
            if Cost[i] <= Cost:
                BestSol = Position(i)
        # Next Mongoose Position
        newtau = np.mean(sm)
        for i in np.arange(1, nScout + 1).reshape(-1):
            M = (np.multiply(Position[i], sm[i])) / Position[i]
            if newtau > tau:
                newPosition = Position[i] - np.multiply(CF * phi * np.random.rand(), (Position[i] - M))
            else:
                newPosition = Position[i] + np.multiply(CF * phi * np.random.rand(), (Position[i] - M))
            tau = newtau
        # Update Best Solution Ever Found
        for i in np.arange(1, nAlphaGroup + 1).reshape(-1):
            if Cost[i] <= Cost:
                BestSol = Position(i)
        # Store Best Cost Ever Found
        BestCost[it] = Cost
        BEF = Cost
        BEP =Position
    ct = time.time()-ct
    return BEP,BestCost,BEF,ct