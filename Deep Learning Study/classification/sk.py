"""
Classifications with classifiers from scikit package

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import feature_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn import tree
from sklearn.cross_validation import train_test_split
import comp5318_assignment1_accuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
import comp5318_assignment1_evaluation as cae

# load dataset
data_set = pd.read_csv('letter-recognition.csv')

letter_set = list(set(data_set.iloc[:, -1]))
letters = np.array([letter_set.index(x) for x in data_set.iloc[:, -1]])

# plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(letter_set))
    plt.xticks(tick_marks, letter_set, rotation=45)
    plt.yticks(tick_marks, letter_set)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# calculate the roc area
n_classes = len(letter_set)
def calculate_roc(truth, predictions):
    lb_truth = label_binarize(truth.iloc[:, -1].astype(int), np.arange(n_classes))
    lb_prediction = label_binarize(predictions.iloc[:, -1].astype(int), np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(letter_set)):
        fpr[i], tpr[i], _ = roc_curve(lb_truth[:, i], lb_prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(lb_truth.ravel(), lb_prediction.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc

# draw the roc area
def draw_roc(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         linewidth=2)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

s_v_m = SVC(kernel='rbf')
svm_kfold = comp5318_assignment1_accuracy.Simple_KFold(10, s_v_m, data_set.iloc[:, :-1].as_matrix(), letters)
print('svm accuracy', np.mean(svm_kfold.calculate()))

svm_fpr, svm_tpr, svm_roc = calculate_roc(svm_kfold.truth, svm_kfold.predictions)
draw_roc(svm_fpr, svm_tpr, svm_roc)

# print the svm precision, recall, and f-score, roc area
print('{},{},{},{},{}'.format('letter', 'precision', 'recall', 'f-score', 'roc'))
for x in letter_set:
    print('{},{:.3f}{:.3f}{:.3f}{:.3f}'.format(x, cae.calculate_precision_one_class(svm_kfold.truth, svm_kfold.predictions, letter_set.index(x)),
                                        cae.calculate_recall_one_class(svm_kfold.truth, svm_kfold.predictions, letter_set.index(x)),
                                        cae.calculate_F_score_one_class(svm_kfold.truth, svm_kfold.predictions, letter_set.index(x)),
                                              svm_roc[letter_set.index(x)]))
print('{},{:.3f}{:.3f}{:.3f}{:.3f}'.format('overall', cae.calculate_precision_overall(svm_kfold.truth, svm_kfold.predictions),
                                          cae.calculate_recall_overall(svm_kfold.truth, svm_kfold.predictions),
                                          cae.calculate_F_score_overall(svm_kfold.truth, svm_kfold.predictions),
                                          svm_roc['macro']))


c = confusion_matrix(svm_kfold.truth.iloc[:, -1], svm_kfold.predictions.iloc[:, -1])
c_normalized = c.astype('float') / c.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(c_normalized)

print()
# decision tree
dt = DecisionTreeClassifier(max_depth=20)
dt_kfold = comp5318_assignment1_accuracy.Simple_KFold(10, dt, data_set.iloc[:, :-1].as_matrix(), letters)
print('decision tree accuracy', np.mean(dt_kfold.calculate()))

print()
# random forest
rf = ensemble.RandomForestClassifier(n_estimators=100, max_features=0.5, max_depth=20)
rf_kfold = comp5318_assignment1_accuracy.Simple_KFold(10, rf, data_set.iloc[:, :-1].as_matrix(), letters)
print('random forest accuracy', np.mean(rf_kfold.calculate()))

print()
# adaboost with decision tree
ab = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=14),
            n_estimators=200,
            learning_rate=1)
ab_kfold = comp5318_assignment1_accuracy.Simple_KFold(10, ab, data_set.iloc[:, :-1].as_matrix(), letters)
print('adaboost accuracy', np.mean(ab_kfold.calculate()))

print()
# bagging with decision tree
bg = ensemble.BaggingClassifier(DecisionTreeClassifier(max_depth=14),
            n_estimators=200)
bg_kfold = comp5318_assignment1_accuracy.Simple_KFold(10, bg, data_set.iloc[:, :-1].as_matrix(), letters)
print('bagging accuracy', bg_kfold.calculate())

print()
print('performance comparison with precision, recall and f-score')
# print performance
print('classifier,precision,recall,f_score')
print('{},{:.3f},{:.3f},{:.3f}'.format(
        'svm',
        cae.calculate_precision_overall(svm_kfold.truth, svm_kfold.predictions),
        cae.calculate_recall_overall(svm_kfold.truth, svm_kfold.predictions),
        cae.calculate_F_score_overall(svm_kfold.truth, svm_kfold.predictions)
    )
)
print('{},{:.3f},{:.3f},{:.3f}'.format(
        'Decision Tree',
        cae.calculate_precision_overall(dt_kfold.truth, dt_kfold.predictions),
        cae.calculate_recall_overall(dt_kfold.truth, dt_kfold.predictions),
        cae.calculate_F_score_overall(dt_kfold.truth, dt_kfold.predictions)
    )
)
print('{},{:.3f},{:.3f},{:.3f}'.format(
        'Random Forest',
        cae.calculate_precision_overall(rf_kfold.truth, rf_kfold.predictions),
        cae.calculate_recall_overall(rf_kfold.truth, rf_kfold.predictions),
        cae.calculate_F_score_overall(rf_kfold.truth, rf_kfold.predictions)
    )
)
print('{},{:.3f},{:.3f},{:.3f}'.format(
        'Adaboost with decision tree',
        cae.calculate_precision_overall(ab_kfold.truth, ab_kfold.predictions),
        cae.calculate_recall_overall(ab_kfold.truth, ab_kfold.predictions),
        cae.calculate_F_score_overall(ab_kfold.truth, ab_kfold.predictions)
    )
)
print('{},{:.3f},{:.3f},{:.3f}'.format(
        'Bagging with decision tree',
        cae.calculate_precision_overall(bg_kfold.truth, bg_kfold.predictions),
        cae.calculate_recall_overall(bg_kfold.truth, bg_kfold.predictions),
        cae.calculate_F_score_overall(bg_kfold.truth, bg_kfold.predictions)
    )
)
