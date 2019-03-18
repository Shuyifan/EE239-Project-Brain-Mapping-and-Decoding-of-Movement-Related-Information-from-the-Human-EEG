import numpy as np
from sklearn.preprocessing import OneHotEncoder

def one_hot(Y):
    Y.shape = (Y.shape[0], 1)
    enc = OneHotEncoder()
    return enc.fit_transform(Y).toarray()

def load_data():
    X_test = np.load("Data/X_test.npy")
    y_test = np.load("Data/y_test.npy")
    person_train_valid = np.load("Data/person_train_valid.npy")
    X_train_valid = np.load("Data/X_train_valid.npy")
    y_train_valid = np.load("Data/y_train_valid.npy")
    person_test = np.load("Data/person_test.npy")

    X_train_valid = X_train_valid[ : , : , : , np.newaxis]
    X_test = X_test[ : , : , : , np.newaxis]
    y_train_valid = one_hot(y_train_valid)
    y_test = one_hot(y_test)
    
    print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
    print ('Test data shape: {}'.format(X_test.shape))
    print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
    print ('Test target shape: {}'.format(y_test.shape))
    print ('Person train/valid shape: {}'.format(person_train_valid.shape))
    print ('Person test shape: {}'.format(person_test.shape))

    print(y_train_valid)

    return [X_train_valid.shape[0], X_test.shape[0], X_train_valid, y_train_valid, X_test, y_test]

def load_data_no_EOG(load_all=None, people=None):
    X_test = np.load("Data/X_test.npy")
    y_test = np.load("Data/y_test.npy")
    person_train_valid = np.load("Data/person_train_valid.npy")
    X_train_valid = np.load("Data/X_train_valid.npy")
    y_train_valid = np.load("Data/y_train_valid.npy")
    person_test = np.load("Data/person_test.npy")

    X_train_valid = X_train_valid[ : , 0 : 22, : , np.newaxis]
    X_test = X_test[ : , 0 : 22, : , np.newaxis]
    y_train_valid = one_hot(y_train_valid)
    y_test = one_hot(y_test)

    if(load_all == None):
        print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
        print ('Test data shape: {}'.format(X_test.shape))
        print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
        print ('Test target shape: {}'.format(y_test.shape))
        print ('Person train/valid shape: {}'.format(person_train_valid.shape))
        print ('Person test shape: {}'.format(person_test.shape))
        return [X_train_valid.shape[0], X_test.shape[0], X_train_valid, y_train_valid, X_test, y_test]
    else:
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
        return [X_train_valid.shape[0], X_test.shape[0], X_train_valid, y_train_valid, X_test, y_test]



def load_data_with_fq_domain():
    X_test = np.load("Data/X_test.npy")
    y_test = np.load("Data/y_test.npy")
    person_train_valid = np.load("Data/person_train_valid.npy")
    X_train_valid = np.load("Data/X_train_valid.npy")
    y_train_valid = np.load("Data/y_train_valid.npy")
    person_test = np.load("Data/person_test.npy")

    x_train_freq = np.zeros((X_train_valid.shape[0], 44, X_train_valid.shape[2]))
    x_test_freq = np.zeros((X_test.shape[0], 44, X_test.shape[2]))
    for i in range(x_train_freq.shape[0]):
        for j in range(44):
            if(j < 22): 
                x_train_freq[i, j, : ] = X_train_valid[i, j, : ]
            else:
                x_train_freq[i, j, : ] = np.absolute(np.fft.fft(X_train_valid[i, j - 22, : ]))

    for i in range(x_test_freq.shape[0]):
        for j in range(44):
            if(j < 22): 
                x_test_freq[i, j, : ] = X_test[i, j, : ]
            else:
                x_test_freq[i, j, : ] = np.absolute(np.fft.fft(X_test[i, j - 22, : ]))

    X_train_valid = x_train_freq[ : , 0 : 44, : , np.newaxis]
    X_test = x_test_freq[ : , 0 : 44, : , np.newaxis]
    y_train_valid = one_hot(y_train_valid)
    y_test = one_hot(y_test)

    print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
    print ('Test data shape: {}'.format(X_test.shape))
    print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
    print ('Test target shape: {}'.format(y_test.shape))
    print ('Person train/valid shape: {}'.format(person_train_valid.shape))
    print ('Person test shape: {}'.format(person_test.shape))

    return [X_train_valid.shape[0], X_test.shape[0], X_train_valid, y_train_valid, X_test, y_test]