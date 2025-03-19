import numpy as np
from Global_Vars import Global_Vars
from Model_AA_GCN_GRU import Model_AA_GCN_GRU


# from Model_ADeepCRF import Model_ADeepCRF
def Objfun(Soln):
    Feat = Global_Vars.Feat
    Target  = Global_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        Eval = Model_AA_GCN_GRU(Feat, Target,sol)
        Fitn[i] =(1 /Eval[13] ) + Eval[11] # Maximization of MCC and Minimization of FDR
    return Fitn

