import numpy as np
import os
import cv2 as cv
from numpy import matlib
from sklearn.utils import shuffle
import random as rn
from DMO import DMO
from Global_Vars import Global_Vars
from LOA import LOA
from Model_AA_GCN_GRU import Model_AA_GCN_GRU
from Model_GCN import Model_GCN
from Model_GRU import Model_GRU
from Model_VGG16 import Model_VGG16
from NRO import NRO
from PROPOSED import PROPOSED
from Plot_Results import Plot_Epoch_Table, Plot_Kfold, plot_roc, Plot_Epoch, plot_results_conv, Confusion_matrix
from WOA import WOA
from objective_function import Objfun

# Read Dataset1
an = 0
if an == 1:
    dir = './Dataset/Dataset1/train/'
    dir_list = os.listdir(dir)
    images = []
    Label = []
    for i in range(len(dir_list)):
        file = dir + dir_list[i] + '/'
        dir_list1 = os.listdir(file)
        for j in range(len(dir_list1)):
            file1 = file + dir_list1[j]
            read = cv.imread(file1)
            read = cv.resize(read, [256, 256])
            filename = dir_list[i]
            images.append(read)
            Label.append(filename)
    Target = np.asarray(Label)
    Uni = np.unique(Target)
    Targ = np.zeros((len(Target), len(Uni))).astype('int')
    for k in range(len(Uni)):
        ind = np.where(Target == Uni[k])
        Targ[ind[0], k] = 1
    images, Targ = shuffle(np.asarray(images), Targ)
    np.save('Images_1.npy', images)
    np.save('Target_1.npy', Targ)

# Read Dataset2
an = 0
if an == 1:
    dir = './Dataset/Dataset2/'
    dir_list = os.listdir(dir)
    images = []
    Label = []
    for i in range(len(dir_list)):
        file = dir + dir_list[i] + '/'
        dir_list1 = os.listdir(file)
        for j in range(len(dir_list1)):
            file1 = file + dir_list1[j]
            read = cv.imread(file1)
            read = cv.resize(read, [256, 256])
            filename = dir_list[i]
            images.append(read)
            Label.append(filename)
    Target = np.asarray(Label)
    Uni = np.unique(Target)
    Targ = np.zeros((len(Target), len(Uni))).astype('int')
    for k in range(len(Uni)):
        ind = np.where(Target == Uni[k])
        Targ[ind[0], k] = 1
    images, Targ = shuffle(np.asarray(images), Targ)
    np.save('Images_2.npy', images)
    np.save('Target_2.npy', Targ)

# Optimization for Classification
an = 0
if an == 1:
    Best = []
    Fit = []
    for a in range(2):
        Data = np.load('Images_' + str(a + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        Global_Vars.Feat = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3
        xmin = matlib.repmat([5,5, 0.01], Npop, 1)
        xmax = matlib.repmat([255,50, 0.99], Npop, 1)
        fname = Objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.asarray(rn.uniform(xmin[p1, p2], xmax[p1, p2]))
        Max_iter = 50

        print("DMO...")
        [bestfit1, fitness1, bestsol1, time1] = DMO(initsol, fname, xmin, xmax, Max_iter)

        print("LOA...")
        [bestfit2, fitness2, bestsol2, time2] = LOA(initsol, fname, xmin, xmax, Max_iter)

        print("NRO...")
        [bestfit4, fitness4, bestsol4, time3] = NRO(initsol, fname, xmin, xmax, Max_iter)

        print("WOA...")
        [bestfit3, fitness3, bestsol3, time4] = WOA(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        Bestsol = ([bestsol1,bestsol2,bestsol3,bestsol4,bestsol5])
        Fitness = ([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])
        Best.append(Bestsol)
        Fit.append(Fitness)
    np.save('Bestsol.npy', np.asarray(Best))
    np.save('Fitness.npy', np.asarray(Fit))

# Classification (Varying Epochs)
an = 0
if an == 1:
    Eval_all = []
    for a in range(2):
        Feat = np.load('Images_' + str(a + 1) + '.npy', allow_pickle=True)
        Bstsol = np.load('Bestsol.npy', allow_pickle=True)[a]
        Target = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        EVAL = []
        Epochs = [100,200,300,400,500]
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        for batch in range(len(Epochs)):
            Eval = np.zeros((10, 25))
            for i in range(len(Bstsol)):
                sol = Bstsol[i, :]
                Eval[i, :] = Model_AA_GCN_GRU(Feat, Target, sol)
            Eval[5:] = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :] = Model_GCN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :] = Model_GRU(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :] = Model_AA_GCN_GRU(Feat, Target)
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Eval_Batch.npy', np.asarray(Eval_all))

Plot_Epoch_Table()
Plot_Kfold()
plot_roc()
Plot_Epoch()
plot_results_conv()
Confusion_matrix()