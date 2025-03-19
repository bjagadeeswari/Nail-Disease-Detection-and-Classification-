import numpy as np
from prettytable import PrettyTable
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn import metrics


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_roc():
    lw = 2
    cls = ['VGG16', 'GCN', 'GRU', 'AAGCN-GRU', 'MWOA-AAGCN-GRU']
    colors = cycle(["m", "b", "r", "lime", "k"])

    Predicted = np.load('roc_score.npy', allow_pickle=True)
    Actual = np.load('roc_act.npy', allow_pickle=True)

    for i in range(len(Actual)):
        Dataset = ['Dataset 1', 'Dataset 2']
        plt.figure(figsize=(8, 6))  # Set figure size

        for j, color in zip(range(5), colors):
            false_positive_rate1, true_positive_rate1, _ = roc_curve(Actual[i, 3, j], Predicted[i, 3, j])
            auc_score = metrics.roc_auc_score(Actual[i, 3, j], Predicted[i, 3, j])

            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=f"{cls[j]} (AUC = {auc_score:.3f})"  # Include AUC in the legend
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)  # Diagonal reference line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title('ROC Curve - ' + str(i + 1))
        plt.legend(loc="lower right")

        path = f"./Results/{Dataset[i]}_Roc_.png"
        plt.savefig(path, dpi=300)  # Save with high resolution
        plt.show()


def Plot_Epoch_Table():
    eval = np.load('Eval_ALL.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'pt',
             'G means', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Table_Terms = [0, 1, 2, 3, 4, 8, 9, 12, 14, 15]
    Algorithm = ['TERMS', 'DMO-AAGCN-GRU', 'LOA-AAGCN-GRU', 'NRO-AAGCN-GRU', 'WOA-AAGCN-GRU', 'MWOA-AAGCN-GRU']
    Classifier = ['TERMS', 'VGG16', 'GCN', 'GRU', 'AAGCN-GRU', 'MWOA-AAGCN-GRU']
    Activation = ['1', '2', '3', '4', '5']
    Dataset = ['Dataset 1', 'Dataset 2']

    for i in range(2):
        value = eval[i, 3, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Terms)])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Table_Terms])
        print('-------------------------------------------------- ', Dataset[i], 'Epochs ',
              'Algorithm Comparison --------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Terms)])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Terms])
        print('-------------------------------------------------- ', Dataset[i], 'Epochs ',
              'Classifier Comparison --------------------------------------------------')
        print(Table)


def Plot_Kfold():
    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'pt',
             'G means', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Term = [0]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[n, k, l, Graph_Term[j] + 4]

            length = np.arange(5)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

            ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='>', markerfacecolor='red',
                    markersize=12,
                    label='DMO-AAGCN-GRU')
            ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='>', markerfacecolor='green',
                    markersize=12,
                    label='LOA-AAGCN-GRU')
            ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='>', markerfacecolor='cyan',
                    markersize=12,
                    label='NRO-AAGCN-GRU')
            ax.plot(length, Graph[:, 3], color='#f504c9', linewidth=3, marker='>', markerfacecolor='#fdff38',
                    markersize=12,
                    label='WOA-AAGCN-GRU')
            ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='>', markerfacecolor='w', markersize=12,
                    label='MWOA-AAGCN-GRU')
            plt.xticks(length, ('1', '2', '3', '4', '5'))
            plt.xlabel('k Fold', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path = "./Results/Dataset_%s_kfold_%s_Alg.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], edgecolor='k', hatch='//', color='r', width=0.10, label="VGG16")
            ax.bar(X + 0.10, Graph[:, 6], edgecolor='k', hatch='-', color='#6dedfd', width=0.10, label="GCN")
            ax.bar(X + 0.20, Graph[:, 7], edgecolor='k', hatch='//', color='lime', width=0.10, label="GRU")
            ax.bar(X + 0.30, Graph[:, 8], edgecolor='k', hatch='-', color='#ed0dd9', width=0.10, label="AAGCN-GRU")
            ax.bar(X + 0.40, Graph[:, 9], edgecolor='w', hatch='..', color='k', width=0.10, label="MWOA-AAGCN-GRU")
            plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
            plt.xlabel('K - Fold')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/Dataset_%s_kfold_%s_Med.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()


def Plot_Epoch():
    eval = np.load('Eval_ALL.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'pt',
             'G means', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Term = [2, 3, 4, 9, 12, 16]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[n, k, l, Graph_Term[j] + 4]

            length = np.arange(5)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

            ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='>', markerfacecolor='red',
                    markersize=12,
                    label='DMO-AAGCN-GRU')
            ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='>', markerfacecolor='green',
                    markersize=12,
                    label='LOA-AAGCN-GRU')
            ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='>', markerfacecolor='cyan',
                    markersize=12,
                    label='NRO-AAGCN-GRU')
            ax.plot(length, Graph[:, 3], color='#f504c9', linewidth=3, marker='>', markerfacecolor='#fdff38',
                    markersize=12,
                    label='WOA-AAGCN-GRU')
            ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='>', markerfacecolor='w', markersize=12,
                    label='MWOA-AAGCN-GRU')

            ax.fill_between(length, Graph[:, 0], Graph[:, 3], color='#acc2d9', alpha=.5)  # ff8400
            ax.fill_between(length, Graph[:, 3], Graph[:, 2], color='#c48efd', alpha=.5)  # 19abff
            ax.fill_between(length, Graph[:, 2], Graph[:, 1], color='#be03fd', alpha=.5)  # 00f7ff
            ax.fill_between(length, Graph[:, 1], Graph[:, 4], color='#b2fba5', alpha=.5)  # ecfc5b
            plt.xticks(length, ('100', '200', '300', '400', '500'))
            plt.xlabel('Epochs', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path = "./Results/Dataset_%s_Epoch_%s_Alg.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], edgecolor='k', hatch='//', color='b', width=0.10, label="VGG16")
            ax.bar(X + 0.10, Graph[:, 6], edgecolor='k', hatch='-', color='#6dedfd', width=0.10, label="GCN")
            ax.bar(X + 0.20, Graph[:, 7], edgecolor='k', hatch='//', color='lime', width=0.10, label="GRU")
            ax.bar(X + 0.30, Graph[:, 8], edgecolor='k', hatch='-', color='#ed0dd9', width=0.10, label="AAGCN-GRU")
            ax.bar(X + 0.40, Graph[:, 9], edgecolor='w', hatch='..', color='k', width=0.10, label="MWOA-AAGCN-GRU")
            plt.xticks(X + 0.25, ('100', '200', '300', '400', '500'))
            plt.xlabel('Epochs')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/Dataset_%s_Epoch_%s_Med.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()


def plot_results_conv():
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'DMO-AAGCN-GRU', 'LOA-AAGCN-GRU', 'NRO-AAGCN-GRU', 'WOA-AAGCN-GRU', 'MWOA-AAGCN-GRU']
    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Dataset = ['Dataset1', 'Dataset2']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- ', Dataset[i], 'Statistical Report ',
              '--------------------------------------------------')
        print(Table)
        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='.', markerfacecolor='red', markersize=12,
                 label='DMO-AAGCN-GRU')
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='.', markerfacecolor='green',
                 markersize=12,
                 label='LOA-AAGCN-GRU')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='.', markerfacecolor='cyan',
                 markersize=12,
                 label='NRO-AAGCN-GRU')
        plt.plot(length, Conv_Graph[3, :], color='y', linewidth=3, marker='.', markerfacecolor='magenta',
                 markersize=12,
                 label='WOA-AAGCN-GRU')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='.', markerfacecolor='black',
                 markersize=12,
                 label='MWOA-AAGCN-GRU')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/%s_Conv.png" % (Dataset[i]))
        plt.show()


def Confusion_matrix():
    for a in range(2):
        Actual = np.load('Actual_' + str(a + 1) + '.npy', allow_pickle=True)
        Predict = np.load('Predict_' + str(a + 1) + '.npy', allow_pickle=True)
        ax = plt.subplot()
        cm = confusion_matrix(np.asarray(Actual[4]).argmax(axis=1), np.asarray(Predict[4]).argmax(axis=1))
        sn.heatmap(cm, annot=True, fmt='g',
                   ax=ax).set(title='Confusion Matrix ' + str(a + 1))
        path = "./Results/Confusion_%s.png" % (str(a + 1))
        plt.savefig(path)
        plt.show()


if __name__ == '__main__':
    Plot_Epoch_Table()
    Plot_Kfold()
    plot_roc()
    Plot_Epoch()
    plot_results_conv()
    Confusion_matrix()
