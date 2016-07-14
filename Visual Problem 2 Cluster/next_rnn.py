from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import script.predict.preprocess as pp
from sklearn import ensemble, cross_validation

import numpy as np
from sklearn import metrics
import pandas

import tensorflow as tf
from tensorflow.contrib import learn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('test_with_fake_data', False,
                         'Test the example code with fake data.')

MAX_DOCUMENT_LENGTH = 45
EMBEDDING_SIZE = 50
n_words = 1692 + 189


def input_op_fn(x):
  """Customized function to transform batched x into embeddings."""
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  word_vectors = learn.ops.categorical_variable(x, n_classes=n_words,
      embedding_size=EMBEDDING_SIZE, name='words')
  # Split into list of embedding per word, while removing doc length dim.
  # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
  word_list = learn.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
  return word_list


def plot_next_place(prev, next, ids, ax=None, max_size=None):
    if ax is None:
        fig, ax = plt.subplots()

    kp = data.read_key_points().set_index('place_id')
    group_info = data.read_group_info('Fri').set_index('group_id')

    # places = pd.DataFrame(data={'prev': prev, 'next': next}).dropna().astype('int64')
    places = pd.DataFrame(data={'prev': prev, 'next': next}, index=ids)
    # drop any rows with 0 for the place id, as we can't plot that.
    places = places.loc[(places != 0).all(axis=1)]
    places['size'] = group_info['size']
    p2 = places.groupby(['next', 'prev']).sum().reset_index().sort_values('size')
    # remove the small slices
    # p2 = p2[p2['size'] >= 8]
    if max_size is None:
        max_size = p2['size'].max()
        # print(max_size)

    im = data.read_image('Grey')
    ax.imshow(im, extent=[0, 100, 0, 100])

    cmap = plt.get_cmap('plasma')
    for i, row in enumerate(p2.itertuples()):
        # index_amt = i / (len(p2) - 1)
        size_amt = row.size / max_size
        prev_xy = kp.loc[row.prev, ['X', 'Y']].values
        next_xy = kp.loc[row.next, ['X', 'Y']].values
        arrowprops = {'arrowstyle': 'simple',
                      'mutation_scale': 50 * size_amt,
                      'alpha': 0.2 + 0.8 * size_amt,
                      'lw': 0,
                      'color': cmap(0.5 * size_amt),
                      'connectionstyle': "arc3,rad=-0.1"}
        ax.annotate('', xy=next_xy, xytext=prev_xy, arrowprops=arrowprops)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])


def get_max_move_size(prev, next, ids):
    group_info = data.read_group_info('Fri').set_index('group_id')
    # places = pd.DataFrame(data={'prev': prev, 'next': next}).dropna().astype('int64')
    places = pd.DataFrame(data={'prev': prev, 'next': next}, index=ids)
    # drop any rows with 0 for the place id, as we can't plot that.
    places = places.loc[(places != 0).all(axis=1)]
    places['size'] = group_info['size']
    p2 = places.groupby(['next', 'prev']).sum().reset_index().sort_values('size')
    # remove the small slices
    # p2 = p2[p2['size'] >= 8]
    max_size = p2['size'].max()
    return max_size


def plot_next_place_probs(prev, probs, places, ids, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    kp = data.read_key_points().set_index('place_id')
    group_info = data.read_group_info('Fri').set_index('group_id')

    df = pd.DataFrame(probs, index=ids, columns=places)
    df2 = df.stack(0).reset_index(level=1)

    # places = pd.DataFrame(data={'prev': prev, 'next': next}).dropna().astype('int64')
    places = pd.DataFrame(data={'prev': prev, 'next': next}, index=ids)
    # drop any rows with 0 for the place id, as we can't plot that.
    places = places.loc[(places != 0).all(axis=1)]
    places['size'] = group_info['size']
    p2 = places.groupby(['next', 'prev']).sum().reset_index().sort_values('size')
    # remove the small slices
    # p2 = p2[p2['size'] >= 8]
    max_size = p2['size'].max()
    print(max_size)

    im = data.read_image('grey')
    ax.imshow(im, extent=[0, 100, 0, 100])

    cmap = plt.get_cmap('plasma')
    for i, row in enumerate(p2.itertuples()):
        # index_amt = i / (len(p2) - 1)
        size_amt = row.size / max_size
        prev_xy = kp.loc[row.prev, ['X', 'Y']].values
        next_xy = kp.loc[row.next, ['X', 'Y']].values
        arrowprops = {'arrowstyle': 'simple',
                      'mutation_scale': 50 * size_amt,
                      'alpha': size_amt,
                      'lw': 0,
                      'color': cmap(0.5 * size_amt),
                      'connectionstyle': "arc3,rad=-0.1"}
        ax.annotate('', xy=next_xy, xytext=prev_xy, arrowprops=arrowprops)


def main():
    # df = data.read_visited_key_points('Fri', grouped=True, extra=['category'])
    # categories = ['Thrill Rides', 'Kiddie Rides', 'Rides for Everyone', 'Shows & Entertainment', 'Shopping']
    # df = df[df['category'].isin(categories)].sort_values('Timestamp')
    #
    # prev = df[df['Timestamp'] <= '2014-06-06 12'].groupby('group_id').last()
    # next = df[df['Timestamp'] > '2014-06-06 12'].groupby('group_id').first()

    categories = ['Thrill Rides', 'Kiddie Rides', 'Rides for Everyone', 'Shows & Entertainment', 'Shopping']
    x, y, prev, ids = pp.get_bag_data(['Fri'], 14, categories, return_prev=True, return_ids=True)
    # discard the day data because we only have 1 day
    ids = ids['group_id'].values
    # clamp x values to 1 or 0
    x = (x > 0).astype('int64')

    x_train, x_test, y_train, y_test, prev_train, prev_test, ids_train, ids_test = (
        cross_validation.train_test_split(x, y, prev, ids, train_size=0.25, random_state=2294967295)
    )

    print('Predicting')

    #################################random forest##################################
    predictor = ensemble.RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    predictor.fit(x_train, y_train)
    y_pred1 = predictor.predict(x_test)

    ################################RNN##################################
    # Build model: a single direction GRU with a single layer
    classifier = learn.TensorFlowRNNClassifier(
        rnn_size=EMBEDDING_SIZE, n_classes=82, cell_type='gru',
        input_op_fn=input_op_fn, num_layers=1, bidirectional=False,
        sequence_length=None, steps=1000, optimizer='Adam',
        learning_rate=0.01, continue_training=True)

    # print(x_train)
    # print(y_train)
    # Train and predict
    classifier.fit(x_train, y_train, steps=1000)
    y_pred2 = classifier.predict(x_test)

    print('Plotting')
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    axs = [ax1, ax2, ax3]

    # sizes = [get_max_move_size(prev_test, y, ids_test) for y in [y_test, y_pred]]
    # max_size = max(sizes)

    axs[0].set_title('Actual Data')
    plot_next_place(prev_test, y_test, ids_test, ax=axs[0])

    axs[1].set_title('RNN Predicted')
    plot_next_place(prev_test, y_pred2, ids_test, ax=axs[1])

    axs[2].set_title('RF Predicted')
    plot_next_place(prev_test, y_pred1, ids_test, ax=axs[2])

    fig1.savefig('actual.png', tight=True)
    fig2.savefig('RNN predicted.png', tight=True)

    plt.show()


if __name__ == '__main__':
    main()