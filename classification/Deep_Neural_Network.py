import tensorflow.contrib.learn as skflow
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold


def main():
    # read in the input data
    input = pd.read_csv('letter-recognition.csv', header=None)
    data = np.array(input.values[:, 1:])

    # get the input labels
    labels1 = np.array(input.values[:, 0])
    labels = []
    # transform letters to numbers ('A'->0, 'B'->1, and so on)
    for label in labels1:
        labels.append(ord(label) - ord('A'))
    labels = np.array(labels)
    n_classes = 26

    # define the classifier with 3 layers (100 units on each layer) and 2000 steps
    classifier = skflow.TensorFlowDNNClassifier(hidden_units=[100, 100, 100], n_classes=n_classes,
                                                learning_rate=0.05, steps=20000)

    scores = []
    # define the 10-fold cross validation
    skf = StratifiedKFold(labels, n_folds=10)
    for train_index, test_index in skf:
        # get the data and labels for both training set and test set
        train_data = data[train_index]
        train_labels = labels[train_index]
        test_data = data[test_index]
        test_labels = labels[test_index]

        # fit data, compute the score
        classifier.fit(train_data, train_labels)
        score = metrics.accuracy_score(test_labels, classifier.predict(test_data))
        scores.append(score)

    # print out average score
    print(scores)
    print("Accuracy: %.5f%%" % (sum(scores) * 10))


if __name__ == '__main__':
    main()