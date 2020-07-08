import sys
from sklearn import metrics
import matplotlib.pyplot as plt

if __name__ == "__main__":

    truth_file = sys.argv[1]
    prediction_file = sys.argv[2]    
    # truth_file = './data/train.csv'
    # prediction_file = './preds_by_tree_lstm.csv'

    truth = []
    prediction = []

    with open(truth_file, "r", encoding="utf-8") as tf:
        for line in tf.readlines()[1:]:
            # truth.append(int(line.strip().split(',')[2]))
            truth.append(int(line.strip().split(',')[1]))

    with open(prediction_file, "r", encoding="utf-8") as pf:
        for line in pf.readlines()[1:]:
            prediction.append(float(line.strip().split(',')[1]))
            # if float(line.strip().split(',')[1]) > 0.5:
            #     prediction.append(1.0)
            # else:
            #     prediction.append(0.0)
    # print(truth)
    # print(prediction)

    roc_auc = metrics.roc_auc_score(truth, prediction)
    p, r, thr = metrics.precision_recall_curve(truth, prediction)
    prc_auc = metrics.auc(r, p)

    print("roc_auc")
    print(roc_auc)

    print("prc_auc")
    print(prc_auc)


    def plot_roc(labels, predict_prob):
        false_positive_rate,true_positive_rate,thresholds=metrics.roc_curve(labels, predict_prob)
        roc_auc=metrics.auc(false_positive_rate, true_positive_rate)
        plt.title('ROC')
        plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.show()

    def plot_prc(labels, predict_prob):
        plt.figure(1)
        plt.title('Precision/Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        precision, recall, thresholds = metrics.precision_recall_curve(labels, predict_prob)
        plt.figure(1)
        plt.plot(precision, recall)
        plt.show()

    # plot_roc(truth, prediction)
    plot_prc(truth, prediction)