import numpy as np
from sklearn.preprocessing import OneHotEncoder

"""
The function to preprocess the data by use one-hot encoding

:param Y: the input label, wchich will be preprocessed by the one-hot encoding. Its shape should be either (N, ) or N(N, 1)

:type Y: numpy.ndarray

:returns Y_one_hot: return the one-hot encoded label

:rtype Y_one_hot: numpy.ndarray
"""
def one_hot(Y):
    Y.shape = (Y.shape[0], 1)
    enc = OneHotEncoder()
    Y_one_hot = enc.fit_transform(Y).toarray()
    return Y_one_hot

"""
The function to load the data.

:param load_all: if it is True, load all the data. If it is False, load the data of a specific person, which is specify by the parameter 'people'
:param people: if the 'load_all' parameter is False, this parameter specify which person's (from 0 to 8) data we will load.
:param EOG: if it is True, keep the EOG channels (22 : 24); else abandon these channels.
:param frequency: if it is False, just use the origin features. (25 channels if EOG=False, 22 channels if EOG=True).
                  If it is True, we expand the features by add a frequency domain channels for each origin channel. We used DFT to get the frequency domain channel.
                  Thus, if it is channels, the number of channel will be 50 if EOG=False, 44 if EOF=True.

:type load_all: bool
:type people: int
:type EOG: bool
:type frequency: bool

:returns data: the loaded data, which is fitted into a list, whose length is 6.
               data[0] represents the total number of training dataset.
               data[1] represents the total number of test dataset. 
               data[2] represents the features of the training dataset.
               data[3] represents the true label of the training dataset after the one-hot encoding.
               data[4] represents the features of the test dataset.
               data[5] represents the true label of the test dataset after the one-hot encoding.

:rtype data: list
"""
def load_data(load_all=True, people=None, EOG=False, frequency=False):
    X_test = np.load("Data/X_test.npy")
    y_test = np.load("Data/y_test.npy")
    person_train_valid = np.load("Data/person_train_valid.npy")
    X_train_valid = np.load("Data/X_train_valid.npy")
    y_train_valid = np.load("Data/y_train_valid.npy")
    person_test = np.load("Data/person_test.npy")

    # Check whether to keep the EOG channels
    if(EOG):
        X_train_valid = X_train_valid[ : , : , : , np.newaxis]
        X_test = X_test[ : , : , : , np.newaxis]
    else:
        X_train_valid = X_train_valid[ : , 0 : 22, : , np.newaxis]
        X_test = X_test[ : , 0 : 22, : , np.newaxis]
    
    # Check whether to expand the channel by adding frequency domain channels
    if(frequency):
        x_train_freq = np.zeros((X_train_valid.shape[0], 2 * X_train_valid.shape[1], X_train_valid.shape[2], 1))
        x_test_freq = np.zeros((X_test.shape[0], 2 * X_train_valid.shape[1], X_test.shape[2], 1))
        for i in range(x_train_freq.shape[0]):
            for j in range(2 * X_train_valid.shape[1]):
                if(j < X_train_valid.shape[1]): 
                    x_train_freq[i, j, : , 0] = X_train_valid[i, j, : , 0]
                else:
                    x_train_freq[i, j, : , 0] = np.absolute(np.fft.fft(X_train_valid[i, j - X_train_valid.shape[1], : , 0]))

        for i in range(x_test_freq.shape[0]):
            for j in range(2 * X_train_valid.shape[1]):
                if(j < X_train_valid.shape[1]): 
                    x_test_freq[i, j, : ] = X_test[i, j, : ]
                else:
                    x_test_freq[i, j, : ] = np.absolute(np.fft.fft(X_test[i, j - X_train_valid.shape[1], : ]))

        X_train_valid = x_train_freq
        X_test = x_test_freq
    
    y_train_valid = one_hot(y_train_valid)
    y_test = one_hot(y_test)
    
    # Check whether we want all the people's data, or just data of a spcific person
    if(not load_all):
        train_tag = [i for i in range(person_train_valid.shape[0]) if person_train_valid[i] == people]
        test_tag = [i for i in range(person_test.shape[0]) if person_test[i] == people]
        X_train_valid = X_train_valid[train_tag, : , : , : ]
        X_test = X_test[test_tag, : , : , : ]
        y_train_valid = y_train_valid[train_tag, : ]
        y_test = y_test[test_tag, : ]
    
    print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
    print ('Test data shape: {}'.format(X_test.shape))
    print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
    print ('Test target shape: {}'.format(y_test.shape))
    print ('Person train/valid shape: {}'.format(person_train_valid.shape))
    print ('Person test shape: {}'.format(person_test.shape))

    data = [X_train_valid.shape[0], X_test.shape[0], X_train_valid, y_train_valid, X_test, y_test]

    return data