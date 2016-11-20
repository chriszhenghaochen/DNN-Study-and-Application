import numpy as np
import pandas as pd

# shuffle data frame
def shuffle_df(df):
    return df.iloc[np.random.permutation(len(df))]


"""
	KFold, split the data set as K equally distributed subsets.
	1. separate the data set as n categories.
	2. insert each subsets one by one in order to have same distribution
	input: data_set, assume the first is the name, and the last is the category
	output: truth, predictions, accuracy
"""	
class Simple_KFold:
    def __init__(self, k, classifier, X, y):
        self.k = k
        self.classifier = classifier
        self.data_set = pd.DataFrame(X)
        self.data_set['category'] = y
        self.data_set = self.data_set.reset_index()
    
    def calculate(self):
        k = self.k
        accuracy = []
        data_set = self.data_set
        
        classes = np.unique(data_set.iloc[:, -1])
        data_set_classes = []
        for i in classes:
            tmp = data_set.loc[data_set.iloc[:, -1] == i,:]
            tmp = shuffle_df(tmp)
            data_set_classes.append(tmp)

        training_and_test_k = []
        for j in np.arange(k):
            training_and_test_k.append([])

        for i in classes:
            tmp = data_set_classes[i]
            j = 0
            n = 0
            while j < tmp.shape[0]:
                training_and_test_k[n].append(tmp.iloc[j, :].tolist())
                j += 1
                n += 1
                n = n % k

        for i in np.arange(k):
            training_and_test_k[i] = pd.DataFrame(training_and_test_k[i], columns=data_set.columns)
            
            
        self.truth = pd.DataFrame(columns=['name', 'category'])
        self.predictions = pd.DataFrame(columns=['name', 'category'])
        for i in np.arange(self.k):
            testing = training_and_test_k[i]
            training = pd.DataFrame(columns=data_set.columns)
            for j in np.arange(k):
                if j != i:
                    training = pd.concat([training, training_and_test_k[j]])
            self.classifier.fit(training.iloc[:, 1:-1].as_matrix(), training.iloc[:, -1].as_matrix())
            predi = self.classifier.predict(testing.iloc[:, 1:-1].as_matrix())
            acc = sum(predi == testing.iloc[:, -1]) / len(predi)
            accuracy.append(acc)
            
            tru = pd.DataFrame(columns=['name', 'category'])
            tru['name'] = testing.iloc[:, 0]
            tru['category'] = testing.iloc[:, -1]
            self.truth = pd.concat([self.truth, tru])
            pred = pd.DataFrame(columns=['name', 'category'])
            pred['name'] = testing.iloc[:, 0]
            pred['category'] = predi
            self.predictions = pd.concat([self.predictions, pred])
        
        return accuracy
    
def simple_split_train_test(data_set, k):
    data_shuffle = shuffle_df(data_set)
    training = data_shuffle.iloc[0:data_shuffle.shape[0] * (k-1) // k, :]
    testing = data_shuffle.iloc[data_shuffle.shape[0] * (k-1) // k:, :]
    return (training, testing)