from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import tree, naive_bayes, neighbors, dummy, ensemble
import tensorflow as tf
from tensorflow.contrib import learn

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

# Build model: a single direction GRU with a single layer
RNNclassifier = learn.TensorFlowRNNClassifier(
    rnn_size=EMBEDDING_SIZE, n_classes=82, cell_type='gru',
    input_op_fn=input_op_fn, num_layers=1, bidirectional=False,
    sequence_length=None, steps=1000, optimizer='Adam',
    learning_rate=0.01, continue_training=True)

DNNclassifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=82)


# ----------------- Prediction Functions ------------------

all_predictors = {
    # 'Decision Tree':
    #     tree.DecisionTreeClassifier(),
    # 'Gradient Boosting':
    #     ensemble.GradientBoostingClassifier(n_estimators=33, learning_rate=1.0, random_state=0),
    'Random Forest':
        ensemble.RandomForestClassifier(max_depth=2),
    'Adaboost':
        ensemble.AdaBoostClassifier(random_state=0),
    'MultinomialNB':
        naive_bayes.MultinomialNB(),
    # 'GaussianNB': gnb_predict,
    'BernoulliNB':
        naive_bayes.BernoulliNB(),
    # 'KNN':
    #     neighbors.KNeighborsClassifier(n_neighbors=10),
    # 'Random':
    #     dummy.DummyClassifier(strategy='stratified'),
    'Most Frequent':
        dummy.DummyClassifier(strategy='most_frequent'),
    'Uniform':
        dummy.DummyClassifier(strategy='uniform'),

    'RNN':
        RNNclassifier,

    'DNN':
        DNNclassifier
}