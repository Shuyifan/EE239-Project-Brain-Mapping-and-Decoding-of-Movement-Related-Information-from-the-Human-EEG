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