import matplotlib.pyplot as plt

def true_positive_rate(preds, labels, threshold):
    total_positive = sum(labels)
    th_preds = [1 if x > threshold else 0 for x in preds]
    tp_count = sum(a==1 and b==1 for a, b in zip(labels, th_preds))
    return tp_count / total_positive

def false_positive_rate(preds, labels, threshold):
    total_negative = len(labels) - sum(labels)
    th_preds = [1 if x > threshold else 0 for x in preds]
    fp_count = sum(a==0 and b==1 for a, b in zip(labels, th_preds))
    return fp_count / total_negative

def accuracy(preds, labels, threshold):
    total = len(labels)
    th_preds = [1 if x > threshold else 0 for x in preds]
    correct = sum(a==b for a, b in zip(labels, th_preds))
    return correct / total

def auc_score(tprs, fprs):
    area = 0
    for i in range(len(tprs)-1):
        area += 0.5 * (tprs[i] + tprs[i+1]) * (fprs[i] - fprs[i+1])
    return area

def roc_curve(preds, labels, resident_num):
    tpr, fpr = [], []
    best_acc = 0

    thresholds = [-1, 0, 0.33, 0.66, 1]
    for thresh in thresholds:
        tpr.append(true_positive_rate(preds, labels, thresh))
        fpr.append(false_positive_rate(preds, labels, thresh))
        acc = accuracy(preds, labels, thresh)
        if acc > best_acc: best_acc = acc
    
    score = auc_score(tpr, fpr)

    plt.figure(figsize = (18, 12))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], "gray", linestyle="dotted")
    plt.xlabel('False-Positive Rate')
    plt.ylabel('True-Positive Rate')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('Area Under ROC Curve')
    plt.legend((f"AUC = {100*score:.2f}\nACC = {100*best_acc:.2f}", " "), loc="lower right")
    plt.savefig(f"./Plot/ROC Curve/Resident{resident_num}.jpg")
    plt.close()
    return