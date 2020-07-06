from sklearn import metrics

truth_file = './data/train.csv'
prediction_file = './preds_by_tree_lstm.csv'

truth = []
prediction = []

with open(truth_file, "r", encoding="utf-8") as tf:
    for line in tf.readlines()[1:]:
        truth.append(int(line.strip().split(',')[2]))

with open(prediction_file, "r", encoding="utf-8") as pf:
    for line in pf.readlines()[1:]:
        prediction.append(float(line.strip().split(',')[1]))

# print(truth)
# print(prediction)

roc_auc = metrics.roc_auc_score(truth, prediction)
p, r, thr = metrics.precision_recall_curve(truth, prediction)
prc_auc = metrics.auc(r, p)

print("roc_auc")
print(roc_auc)

print("prc_auc")
print(prc_auc)