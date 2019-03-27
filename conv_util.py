import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

"""
The function to get the index (of the whole data) in each batch. The function will return a list, each element in the the list is a list
of index of that batch.

:param batchsize: the number of sample in each batch
:param datalen: the total number of the sample

:type batchsize: int
:type datalen: int

:returns id_list: a list, each element in the the list is a list of index of that batch.

:rtype id_list: list
"""
def get_batch_id(batchsize, datalen):
    batchsize = int(batchsize)
    id_all = np.arange(datalen)
    np.random.shuffle(id_all)   
    id_list = []    
    for i in range(int(datalen / batchsize)):
        id_batch = id_all[int(i * batchsize) : int(i * batchsize) + batchsize]
        id_list.append(id_batch)
    if datalen % batchsize != 0:
        i += 1
        id_batch = id_all[int(i * batchsize):]
        id_list.append(id_batch)
    return id_list

"""
The shallow net with the best parameter we found on the input dataset (without EOG channels, and no frequency domain channel expansion).
The training parameter can be shown as:
    Xavier Initializer Batch Size: 211  Learning rate:  0.005  AdamOptim izer  Decay rate: 0.975 (after epoch > 40)  Epoch: 100 
The architecture can be shown as:
    Input: 2115 * 22 * 1000 * 1 
    Conv (stride: (1, 2), kernel: (1, 40), filter size: 40) 
    Activation Map: 2115 * 22 * 481 * 40 
    Batch Normalization (momentum: 0.1) 
    Conv (stride: (1, 1), kernel: (22, 1), filter size: 40) 
    Activation Map: 2115 * 1 * 481 * 40 
    Batch Normalization (momentum: 0.1) 
    Square Activation 
    Average Pool (stride: (1, 15), kernel: (1, 75)) 
    Activation Map: 2115 * 1 * 28 * 40 
    Dropout (p: 0.3) 
    Fully Connect Output: 2115 * 4 
    Softmax 

:param data_list: the input data, which is a list, whose length is 6.
                  data_list[0] represents the total number of training dataset.
                  data_list[1] represents the total number of test dataset. 
                  data_list[2] represents the features of the training dataset.
                  data_list[3] represents the true label of the training dataset after the one-hot encoding.
                  data_list[4] represents the features of the test dataset.
                  data_list[5] represents the true label of the test dataset after the one-hot encoding.
:param batch_num: the number of batches for each training epoch.
:param lr: learning rate.
:param total_epoch: the total epochs during the training.
:param decay_rate: the decay rate.
:param start_decay_epoch: the epoch when the learning rate starts decaying (with decay rate is equal to parameter 'decay_rate')
:param drop_out: the drop out rate for the drop out layer

:type data_list: list
:type batch_num: int
:type lr: int
:type total_epoch: int
:type decay_rate: int
:type start_decay_epoch: int
:type drop_out: int
"""
def shallow_net(data_list, batch_num=10, lr=0.005, total_epoch=100, decay_rate=0.975, start_decay_epoch=40, drop_out=0.3):
    [train_len, test_len, x_train, y_train, x_test, y_test] = data_list
    
    #Define graph
    gf = tf.Graph()

    with gf.as_default():
        initK = tf.contrib.layers.xavier_initializer()
        #Placeholders
        # dim N * 22 * 1000 * 1
        X  = tf.placeholder(dtype=tf.float32, shape=(None, 22, 1000, 1), name='X')
        Y  = tf.placeholder(dtype=tf.float32, shape=None, name='Y')
        T  = tf.placeholder(tf.bool)
        LR = tf.placeholder(tf.float32)

        H = tf.layers.conv2d(X, filters=40, kernel_size=(1, 40), strides=(1, 2), padding='valid', activation=None , kernel_initializer=initK)
        H = tf.layers.batch_normalization(H, momentum=0.1, training=T) 
        # dim N * 22 * 481 * 40

        H = tf.layers.conv2d(H, filters=40, kernel_size=(22, 1), strides=(1, 1), padding='valid', activation=None, kernel_initializer=initK)
        H = tf.layers.batch_normalization(H, momentum=0.1, training=T) 
        # dim N * 1 * 481 *40

        H = tf.square(H)
        H = tf.layers.average_pooling2d(H, pool_size=(1, 75) , strides=(1, 15), padding='valid')
        # dim N * 1 * 28 * 40
        
        H = tf.log(H)
        H = tf.layers.dropout(H, rate=drop_out, training=T)

        H = tf.layers.conv2d(H, filters=4, kernel_size=(1, 28), padding='valid', activation=None, kernel_initializer=initK)   
        # dim N * 1 * 1 * 4

        Y_pred = tf.reshape(H,[-1, 4], name='Y_pred')
        # dim N * 4

        Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
        Update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(Update_ops):
            Train_step = tf.train.AdamOptimizer(LR).minimize(Loss)

        init2 = tf.global_variables_initializer()
        
        ### training model ###
        batch_size = train_len / batch_num

        loss_hist   = []
        loss_hist2  = []
        train_hist  = []
        test_hist2  = []

        with tf.Session(graph=gf) as se:
            se.run(init2)
            for epoch in range(total_epoch):
                if epoch > start_decay_epoch:
                    lr *= decay_rate
                id_list = get_batch_id(batch_size, train_len)
                for batch_id in id_list:
                    batch_x = x_train[batch_id]
                    batch_y = y_train[batch_id]
                    
                    _ , loss_i = se.run([Train_step,Loss], feed_dict={X: batch_x, Y: batch_y, T: True, LR: lr})

                loss_test = se.run(Loss,feed_dict={X: x_test, Y: y_test, T: False})

                y_test_pred    = se.run(Y_pred, feed_dict={X: x_test, T: False})
                Y_correct_test = tf.equal(tf.argmax(y_test_pred, 1), tf.argmax(y_test, 1))
                Y_correct_test = tf.cast(Y_correct_test, tf.float32)
                acc_tf_test    = tf.reduce_mean(Y_correct_test)             
                acc_test2      = se.run(acc_tf_test)   

                train_loss = 0.0
                train_acc  = 0.0

                for batch_id in id_list:
                    batch_x = x_train[batch_id]
                    batch_y = y_train[batch_id]

                    loss_batch_train = se.run(Loss,feed_dict={X: batch_x, Y: batch_y, T: False})
                    Y_train_pred     = se.run(Y_pred, feed_dict={X: batch_x, Y: batch_y, T: False})
                    Y_correct_train  = tf.equal(tf.argmax(Y_train_pred, 1), tf.argmax(batch_y, 1))
                    Y_correct_train  = tf.cast(Y_correct_train, tf.float32)
                    acc_tf_train     = tf.reduce_mean(Y_correct_train)             
                    acc_train        = se.run(acc_tf_train)
                    train_loss       = train_loss + loss_batch_train * batch_x.shape[0] / x_train.shape[0]
                    train_acc        = train_acc + acc_train * batch_x.shape[0] / x_train.shape[0]

                loss_hist.append(train_loss)
                loss_hist2.append(loss_test)
                train_hist.append(train_acc)
                test_hist2.append(acc_test2)

                print('Epoch:', epoch, '| test: %.4f' % acc_test2, '| train: %.4f' % train_acc,
                      '| train Loss: %.4f' % train_loss, '| test Loss: %.4f' % loss_test)

        plt.figure()
        plt.plot(train_hist, color='red', label='train')
        plt.plot(test_hist2, color='green', label='test')
        
        plt.ylabel('Test_acc , Train_acc')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(loss_hist, color='red', label='train loss')
        plt.plot(loss_hist2, color='green', label='test  loss')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

"""
The deep net with the best parameter we found on the input dataset (without EOG channels, and no frequency domain channel expansion).
The training parameter can be shown as:
    Xavier Initializer Batch Size: 211  Learning rate:  0.005  AdamOptim izer  Decay rate: 0.975 (after epoch > 40)  Epoch: 200
The architecture can be shown as:
    Input: 2115 * 22 * 1000 * 1  
    Conv (stride: (1, 1), kernel: (1, 25), filter size: 40) 
    Activation Map: 2115 * 22 * 976 * 40 
    Batch Normalization (momentum: 0.1) 
    Conv (stride: (1, 1), kernel: (22, 1) filter size: 40) 
    Activation Map: 2115 * 1 * 976 * 40 
    Batch Normalization (momentum: 0.1) 
    Max Pool (stride: (1, 6), kernel: (1, 6)) 
    Activation Map: 2115 * 1 * 162 * 40 
    Conv (stride: (1, 1), kernel: (1, 23) filter size: 40) 
    Activation Map: 2115 * 1 * 140 * 40 
    Batch Normalization (momentum: 0.1) 
    Square Activation 
    Max Pool (stride: (1, 5), kernel: (1, 5)) 
    Activation Map: 2115 * 1 * 28 * 40 
    Dropout (p: 0.6) 
    Fully Connect Output: 2115 * 4 
    Softmax 

:param data_list: the input data, which is a list, whose length is 6.
                  data_list[0] represents the total number of training dataset.
                  data_list[1] represents the total number of test dataset. 
                  data_list[2] represents the features of the training dataset.
                  data_list[3] represents the true label of the training dataset after the one-hot encoding.
                  data_list[4] represents the features of the test dataset.
                  data_list[5] represents the true label of the test dataset after the one-hot encoding.
:param batch_num: the number of batches for each training epoch.
:param lr: learning rate.
:param total_epoch: the total epochs during the training.
:param decay_rate: the decay rate.
:param start_decay_epoch: the epoch when the learning rate starts decaying (with decay rate is equal to parameter 'decay_rate')
:param drop_out: the drop out rate for the drop out layer

:type data_list: list
:type batch_num: int
:type lr: int
:type total_epoch: int
:type decay_rate: int
:type start_decay_epoch: int
:type drop_out: int
"""
def deep_net(data_list, batch_num=10, lr=0.005, total_epoch=200, decay_rate=0.975, start_decay_epoch=40, drop_out=0.6):

    [train_len, test_len, x_train, y_train, x_test, y_test] = data_list
    
    #Define graph
    gf = tf.Graph()

    with gf.as_default():
        initK = tf.contrib.layers.xavier_initializer()
        #Placeholders
        # dim N * 22 * 1000 * 1
        X  = tf.placeholder(dtype=tf.float32, shape=(None, 22, 1000, 1), name='X')
        Y  = tf.placeholder(dtype=tf.float32, shape=None, name='Y')
        T  = tf.placeholder(tf.bool)
        LR = tf.placeholder(tf.float32)

        H = tf.layers.conv2d(X, filters=40, kernel_size=(1, 25), strides=(1, 1), padding='valid', activation=None, kernel_initializer=initK)
        H = tf.layers.batch_normalization(H, momentum=0.1, training=T) 
        # dim N * 22 * 976 * 40

        H = tf.layers.conv2d(H, filters=40, kernel_size=(22, 1), strides=(1, 1), padding='valid', activation=tf.nn.relu, kernel_initializer=initK)
        H = tf.layers.batch_normalization(H, momentum=0.1, training=T) 
        # dim N * 1 * 976 * 40

        H = tf.layers.max_pooling2d(H, pool_size=(1, 6), strides=(1, 6), padding='valid')
        # dim N * 1 * 162 * 40      

        H = tf.layers.conv2d(H,filters=40, kernel_size=(1, 23), strides=(1, 1), padding='valid', activation=tf.nn.relu, kernel_initializer=initK)
        H = tf.layers.batch_normalization(H, momentum=0.1, training=T) 
        # dim N * 1 * 140 * 40      

        H = tf.square(H)
        H = tf.layers.max_pooling2d(H, pool_size=(1, 5), strides=(1, 5), padding='valid')
        # dim N * 1 * 28 * 40

        H = tf.log(H)
        H = tf.layers.dropout(H, rate=drop_out, training=T)

        H = tf.layers.conv2d(H, filters=4, kernel_size=(1, 28), padding='valid', activation=None, kernel_initializer=initK)   
        # dim N * 1 * 1 * 4

        Y_pred = tf.reshape(H,[-1, 4], name='Y_pred')
        # dim N * 4

        Y_pred = tf.nn.softmax(Y_pred, name='Y_pred')
        Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
        Update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(Update_ops):
            Train_step = tf.train.AdamOptimizer(LR).minimize(Loss)

        init2 = tf.global_variables_initializer()
        
        ### training model ###
        batch_size = train_len / batch_num

        loss_hist   = []
        loss_hist2  = []
        train_hist  = []
        test_hist2  = []

        with tf.Session(graph=gf) as se:
            se.run(init2)
            for epoch in range(total_epoch):
                if epoch > start_decay_epoch:
                    lr *= decay_rate
                id_list = get_batch_id(batch_size, train_len)
                for batch_id in id_list:
                    batch_x = x_train[batch_id]
                    batch_y = y_train[batch_id]
                    
                    _ , loss_i = se.run([Train_step,Loss], feed_dict={X: batch_x, Y: batch_y, T: True, LR: lr})

                loss_test = se.run(Loss, feed_dict={X: x_test, Y: y_test, T: False})

                y_test_pred    = se.run(Y_pred, feed_dict={X: x_test, T: False})
                Y_correct_test = tf.equal(tf.argmax(y_test_pred, 1), tf.argmax(y_test, 1))
                Y_correct_test = tf.cast(Y_correct_test, tf.float32)
                acc_tf_test    = tf.reduce_mean(Y_correct_test)             
                acc_test2      = se.run(acc_tf_test)

                train_loss = 0.0
                train_acc = 0.0

                for batch_id in id_list:
                    batch_x = x_train[batch_id]
                    batch_y = y_train[batch_id]

                    loss_batch_train = se.run(Loss, feed_dict={X: batch_x, Y: batch_y, T: False})
                    Y_train_pred     = se.run(Y_pred, feed_dict={X: batch_x, Y: batch_y, T: False})
                    Y_correct_train  = tf.equal(tf.argmax(Y_train_pred, 1), tf.argmax(batch_y, 1))
                    Y_correct_train  = tf.cast(Y_correct_train, tf.float32)
                    acc_tf_train     = tf.reduce_mean(Y_correct_train)             
                    acc_train        = se.run(acc_tf_train)
                    train_loss       = train_loss + loss_batch_train * batch_x.shape[0] / x_train.shape[0]
                    train_acc        = train_acc + acc_train * batch_x.shape[0] / x_train.shape[0]

                loss_hist.append(train_loss)
                loss_hist2.append(loss_test)
                train_hist.append(train_acc)
                test_hist2.append(acc_test2)

                print('Epoch:', epoch, '| test: %.4f' % acc_test2, '| train: %.4f' % train_acc,
                      '| train Loss: %.4f' % train_loss, '| test Loss: %.4f' % loss_test)

        plt.figure()
        plt.plot(train_hist, color='red', label='train')
        plt.plot(test_hist2, color='green', label='test')
        
        plt.ylabel('Test_acc , Train_acc')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(loss_hist, color='red', label='train loss')
        plt.plot(loss_hist2, color='green', label='test  loss')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

"""
The shallow net with the best parameter we found on the input dataset (without EOG channels, but with frequency domain channel expansion).
The training parameter can be shown as:
    Xavier Initializer Batch Size: 105  Learning rate:  0.005  AdamOptim izer  Decay rate: 0.975 (after epoch > 40)  Epoch: 100 
The architecture can be shown as:
    Input: 2115 * 44 * 1000 * 1 
    Conv (stride: (1, 1), kernel: (1, 40), filter size: 40) 
    Batch Normalization (momentum: 0.1) 
    Activation Map: 2115 * 44 * 961 * 40 
    Conv (stride: (1, 1), kernel: (44, 1) filter size: 40) 
    Batch Normalization (momentum: 0.1) 
    Activation Map: 2115 * 1 * 961 * 40 
    Square Activation 
    Average Pool (stride: (1, 15), kernel: (1, 75)) 
    Dropout (p: 0.3) 
    Activation Map: 2115 * 1 * 60 * 40 
    Fully Connect Output: 2115 * 4 
    Softmax

:param data_list: the input data, which is a list, whose length is 6.
                  data_list[0] represents the total number of training dataset.
                  data_list[1] represents the total number of test dataset. 
                  data_list[2] represents the features of the training dataset.
                  data_list[3] represents the true label of the training dataset after the one-hot encoding.
                  data_list[4] represents the features of the test dataset.
                  data_list[5] represents the true label of the test dataset after the one-hot encoding.
:param batch_num: the number of batches for each training epoch.
:param lr: learning rate.
:param total_epoch: the total epochs during the training.
:param decay_rate: the decay rate.
:param start_decay_epoch: the epoch when the learning rate starts decaying (with decay rate is equal to parameter 'decay_rate')
:param drop_out: the drop out rate for the drop out layer

:type data_list: list
:type batch_num: int
:type lr: int
:type total_epoch: int
:type decay_rate: int
:type start_decay_epoch: int
:type drop_out: int
"""
def shallow_net_frequency(data_list, batch_num=20, lr=0.005, total_epoch=100, decay_rate=0.975, start_decay_epoch=40, drop_out=0.3):

    [train_len, test_len, x_train, y_train, x_test, y_test] = data_list
    
    #Define graph
    gf = tf.Graph()

    with gf.as_default():
        initK = tf.contrib.layers.xavier_initializer()
        #Placeholders
        # dim N * 44 * 1000 * 1
        X  = tf.placeholder(dtype=tf.float32, shape=(None, 44, 1000, 1), name='X')
        Y  = tf.placeholder(dtype=tf.float32, shape=None, name='Y')
        T  = tf.placeholder(tf.bool)
        LR = tf.placeholder(tf.float32)

        H = tf.layers.conv2d(X, filters=40, kernel_size=(1, 40), strides=(1, 1), padding='valid', activation=None, kernel_initializer=initK)
        H = tf.layers.batch_normalization(H, momentum=0.1, training=T) 
        # dim N * 44 * 961 * 40

        H = tf.layers.conv2d(H, filters=40, kernel_size=(44, 1), strides=(1, 1), padding='valid', activation=None, kernel_initializer=initK)
        H = tf.layers.batch_normalization(H, momentum=0.1, training=T)
        # dim N * 1 * 961 *40

        H = tf.square(H)
        H = tf.layers.average_pooling2d(H, pool_size=(1, 75), strides=(1, 15), padding='valid')
        # dim N * 1 * 60 * 40
        
        H = tf.log(H)
        H = tf.layers.dropout(H, rate=drop_out, training=T)

        H = tf.layers.conv2d(H, filters=4, kernel_size=(1, 60), padding='valid', activation=None, kernel_initializer=initK)   
        # dim N * 1 * 1 * 4

        Y_pred = tf.reshape(H,[-1, 4], name='Y_pred')
        # dim N * 4

        #Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))
        Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
        Update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(Update_ops):
            Train_step = tf.train.AdamOptimizer(LR).minimize(Loss)

        init2 = tf.global_variables_initializer()
        
        
        ### training model ###
        batch_size = train_len / batch_num

        loss_hist   = []
        loss_hist2  = []
        train_hist  = []
        test_hist2  = []

        with tf.Session(graph=gf) as se:
            se.run(init2)
            for epoch in range(total_epoch):
                if epoch > start_decay_epoch:
                    lr *= decay_rate
                id_list = get_batch_id(batch_size, train_len)
                for batch_id in id_list:
                    batch_x = x_train[batch_id]
                    batch_y = y_train[batch_id]
                    
                    _ , loss_i = se.run([Train_step,Loss], feed_dict={X: batch_x, Y: batch_y, T: True, LR:lr})

                test_batch_id = get_batch_id(test_len / 4, test_len)
                test_loss = 0.0
                test_acc = 0.0
                for batch_id in test_batch_id:
                    batch_x_test = x_test[batch_id]
                    batch_y_test = y_test[batch_id]
                    
                    loss_test = se.run(Loss, feed_dict={X: batch_x_test, Y: batch_y_test, T: False})

                    y_test_pred    = se.run(Y_pred, feed_dict={X: batch_x_test, T: False})
                    Y_correct_test = tf.equal(tf.argmax(y_test_pred, 1), tf.argmax(batch_y_test, 1))
                    Y_correct_test = tf.cast(Y_correct_test, tf.float32)
                    acc_tf_test    = tf.reduce_mean(Y_correct_test)             
                    acc_test2      = se.run(acc_tf_test)   
                    test_loss      = test_loss + loss_test * batch_x_test.shape[0] / x_test.shape[0]
                    test_acc       = test_acc + acc_test2 * batch_x_test.shape[0] / x_test.shape[0]

                train_loss = 0.0
                train_acc = 0.0

                for batch_id in id_list:
                    batch_x = x_train[batch_id]
                    batch_y = y_train[batch_id]

                    loss_batch_train = se.run(Loss, feed_dict={X: batch_x, Y: batch_y, T: False})
                    Y_train_pred     = se.run(Y_pred, feed_dict={X: batch_x, Y: batch_y, T: False})
                    Y_correct_train  = tf.equal(tf.argmax(Y_train_pred, 1), tf.argmax(batch_y, 1))
                    Y_correct_train  = tf.cast(Y_correct_train, tf.float32)
                    acc_tf_train     = tf.reduce_mean(Y_correct_train)             
                    acc_train        = se.run(acc_tf_train)
                    train_loss       = train_loss + loss_batch_train * batch_x.shape[0] / x_train.shape[0]
                    train_acc        = train_acc + acc_train * batch_x.shape[0] / x_train.shape[0]

                loss_hist.append(train_loss)
                loss_hist2.append(loss_test)
                train_hist.append(train_acc)
                test_hist2.append(acc_test2)

                print('Epoch:', epoch, '| test: %.4f' % acc_test2, '| train: %.4f' % train_acc,
                      '| train Loss: %.4f' % train_loss, '| test Loss: %.4f' % loss_test)

        plt.figure()
        plt.plot(train_hist, color='red', label='train')
        plt.plot(test_hist2, color='green', label='test')
        
        plt.ylabel('Test_acc , Train_acc')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(loss_hist, color='red', label='train loss')
        plt.plot(loss_hist2, color='green', label='test  loss')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()