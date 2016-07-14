"""
A variety of functions for evaluating the accuracy of prediction models.

The actual preprocessing and predicting has been moved to the predict and preprocess modules
"""

import data
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
from sklearn import manifold, decomposition, metrics, cross_validation
from script.predict.predictorsrnn import *
from script.predict.preprocess import *


def check_accuracy(y, i_t, probs, categories, n_predictions):
    if n_predictions > probs.shape[1]:
        n_predictions = probs.shape[1]
    # reverse the probabilities, and then take the top n
    # (note that this could be done with one slice but for readability sake I won't)
    top_n_indices = probs.argsort()[:, ::-1][:, :n_predictions]
    top_n_places = categories[top_n_indices]
    # need to do this so it's the right number of dimensions to compare with the top_n_places
    extended_y = np.tile(y, (n_predictions, 1)).T
    # total = np.mean(top_n_places == extended_y) * n_predictions

    top_n_places = np.delete(top_n_places, i_t, axis=0)
    extended_y = np.delete(extended_y, i_t, axis=0)
    untrained = np.mean(top_n_places == extended_y) * n_predictions
    return untrained


def check_total_accuracy(y, i_t, probs, categories):
    """
    This function rates the models based on the chance they assigned to each option
    (rather than just looking at the order or probabilities, like the check_accuracy function does)
    """
    df = pd.DataFrame(probs, columns=categories)
    # add extra columns in case there were values of y that weren't in the training data
    # (this can also remove columns if there's no y values that matches it, but that's ok actually)
    df = df.reindex(columns=np.unique(y)).fillna(0)
    # each row corresponds to the chance that the predictor gives to the actual example.
    # total_sum = df.lookup(np.arange(len(y)), y).sum() / len(y)
    untrained_sum = df.lookup(i_t, y[i_t]).sum() / len(i_t)
    return untrained_sum


# def test_accuracy(x_train, x_test, y_train, y_test, predictor_names, predictors):
#     accuracies = {}
#
#     print('Predicting')
#     for name, function in zip(predictor_names, predictors):
#         print('  {}'.format(name))
#         p = function(x_train, y_train)
#
#         accuracies[name] = {}
#
#
#
#         probs = p.predict_proba(x_test)
#         df = pd.DataFrame(probs, columns=p.classes_)
#         # add extra columns in case there were values of y that weren't in the training data
#         # (this can also remove columns if there's no y values that matches it, but that's ok actually)
#         df = df.reindex(columns=np.unique(y_test)).fillna(0)
#         probs = df.values
#         accuracies[name]['log_loss'] = metrics.log_loss(y_test, probs)
#
#         # probs = p.predict_proba(x)
#         # if total:
#         #     # the results where the average is stored before being calculated
#         #     untrained_accuracy = check_total_accuracy(y, i_t, probs, p.classes_)
#         #     untrained_accuracies[name] = untrained_accuracy
#         # else:
#         #     untrained_accuracies[name] = np.zeros(max_predictions)
#         #
#         #     for i in range(max_predictions):
#         #         # the results where the average is stored before being calculated
#         #         untrained_accuracy = check_accuracy(y, i_t, probs, p.classes_, i+1)
#         #         untrained_accuracies[name][i] = untrained_accuracy
#
#     return accuracies


def get_accuracy_score(predictor, x_test, y_test):
    y_pred = predictor.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred)
    print(score)
    return score


def get_log_loss_score(predictor, x_test, y_test):

    probs = predictor.predict_proba(x_test)
    df = pd.DataFrame(probs, columns=predictor.classes_)
    # add extra columns in case there were values of y that weren't in the training data
    # (this can also remove columns if there's no y values that matches it, but that's ok actually)
    df = df.reindex(columns=np.unique(y_test)).fillna(0)
    probs = df.values
    # print(metrics.log_loss(y_test, probs))
    return metrics.log_loss(y_test, probs)


def plot_data(x, y):
    mds = manifold.MDS(2, max_iter=200, n_init=1, n_jobs=-1)
    x_fit = mds.fit_transform(x)
    df = pd.DataFrame(data={"x0": x_fit[:, 0], "x1": x_fit[:, 1], "y": y})
    plt.figure()
    for (name, group), color in zip(df.groupby('y'), itertools.cycle(data.palette20)):
        plt.scatter(x=group['x0'], y=group['x1'], c=color, lw=0)

    pca = decomposition.PCA(n_components=None, copy=True, whiten=False)
    x_fit = pca.fit_transform(x)
    df = pd.DataFrame(data={"x0": x_fit[:, 0], "x1": x_fit[:, 1], "y": y})
    plt.figure()
    for (name, group), color in zip(df.groupby('y'), itertools.cycle(data.palette20)):
        plt.scatter(x=group['x0'], y=group['x1'], c=color, lw=0)

    plt.show()


def main():
    kp = data.read_key_points()
    categories = kp['category']
    # categories = categories[~categories.isin(['Restrooms', 'Entry/Exit'])].unique()
    categories = ['Thrill Rides', 'Kiddie Rides', 'Rides for Everyone', 'Shows & Entertainment', 'Shopping']
    # predictor_names = ['Decision Tree',
    #                    # 'Gradient Boosting',
    #                    'Random Forest',
    #                    'MultinomialNB',
    #                    'BernoulliNB',
    #                    # 'KNN',
    #                    'Random',
    #                    'Most Frequent',
    #                    'Uniform']
    predictor_names = all_predictors.keys()

    # times = np.arange(9, 22)
    times = timedelta(hours=1) * np.linspace(9, 22, 20)
    accuracies = {}

    for i, cutoff in enumerate(times):
        print('cutoff {}'.format(cutoff))

        # Preprocessing
        print('Preprocessing')
        # x, y = get_sequence_data(['Sat'], cutoff, categories=categories)
        # x = x[:,:2]

        x, y = get_bag_data(['Sat'], cutoff, categories=categories)
        # change the input stuff to be only 1's and 0's
        x = (x > 0).astype('int64')


        x_train, x_test, y_train, y_test = (
            cross_validation.train_test_split(x, y, train_size=0.9, random_state=2294967295)
        )

        # x_train, y_train = get_bag_data(['Fri', 'Sat'], cutoff, categories=categories)
        # x_test, y_test = get_bag_data(['Sun'], cutoff, categories=categories)

        # Predicting
        print('Predicting')
        for name in predictor_names:
            predictor = all_predictors[name]
            if name not in accuracies:
                accuracies[name] = {
                    'accuracy': np.zeros(len(times)),
                    'log_loss': np.zeros(len(times))
                }

            print('  {}'.format(name))

            if(name == 'RNN'):
                #reset the RNN model
                predictor = learn.TensorFlowRNNClassifier(
                    rnn_size=EMBEDDING_SIZE, n_classes=82, cell_type='gru',
                    input_op_fn=input_op_fn, num_layers=1, bidirectional=False,
                    sequence_length=None, steps=1000, optimizer='Adam',
                    learning_rate=0.01, continue_training=True)

                predictor.fit(x_train, y_train, steps=100)
                print("get you RNN")

            elif(name == 'DNN'):
                #reset the DNN model
                predictor = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=82)
                predictor.fit(x_train, y_train, steps=100)
                print("get you DNN")

            else:
                predictor.fit(x_train, y_train)

            accuracies[name]['accuracy'][i] = get_accuracy_score(predictor, x_test, y_test)
            # accuracies[name]['log_loss'][i] = np.exp(-get_log_loss_score(predictor, x_test, y_test))

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('With data of all types')
    colours = data.palette10
    for score, ax in zip(['accuracy', 'log_loss'], axs):
        for name, color in zip(predictor_names, colours):
            # acc = untrained_accuracies[name] - untrained_accuracies['Random']
            acc = accuracies[name][score]

            ax.plot(times / timedelta(hours=1), acc, c=color)  # , marker='o', markeredgewidth=0, markersize=4)
            ax.set_title('Accuracy using {} scoring metric'.format(score))
            ax.set_ylabel(score)
            ax.set_xlabel('Time of day')
            # ax.set_ylim([-0.5, 0.5])
            # ax.set_ylim([0, 1])
        ax.legend([mpatches.Patch(color=colours[i]) for i in range(len(predictor_names))],
                  predictor_names, prop={'size': 8}, loc="best")

    # fig, axs = plt.subplots(2, 3)
    # for cutoff, ax in zip([10, 12, 14, 16, 18, 20], axs.flat):
    #     total_accuracies, untrained_accuracies = test_accuracy(
    #             get_bag_data, predictor_names, predictors, ['Fri', 'Sat', 'Sun'], cutoff, categories)
    #
    #     for name, color in zip(predictor_names, data.palette10):
    #         # acc = untrained_accuracies[name] - untrained_accuracies['Random']
    #         acc = untrained_accuracies[name]
    #         ixs = np.arange(len(acc))
    #         ax.plot(ixs+1, acc, c=color, marker='o', markeredgewidth=0)
    #         ax.set_title('Prediction accuracy at {}:00'.format(cutoff))
    #         ax.set_ylabel('Chance at least one prediction is correct')
    #         ax.set_xlabel('Number of predictions')
    #         # ax.set_ylim([-0.5, 0.5])
    #         ax.set_ylim([0, 1])
    #     ax.legend([mpatches.Patch(color=data.palette10[i]) for i in range(len(predictor_names))],
    #               predictor_names, prop={'size': 8}, loc="best")
    plt.show()


if __name__ == '__main__':
    main()