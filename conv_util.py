import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def get_batch_id(batchsize,datalen):
    batchsize = int(batchsize)
    id_all = np.arange(datalen)
    np.random.shuffle(id_all)   
    id_list = []    
    for i in range(int(datalen/batchsize)):
        id_batch = id_all[int(i*batchsize):int(i*batchsize)+batchsize]
        id_list.append(id_batch)
    if datalen % batchsize !=0:
        i+=1
        id_batch = id_all[int(i*batchsize):]
        id_list.append(id_batch)
    return id_list

def ConvNet_nocrop(data_list):

    [train_len, test_len ,x_train, y_train, x_test , y_test] = data_list
    
    #Define graph
    gf = tf.Graph()

    with gf.as_default():
        initK = tf.contrib.layers.xavier_initializer()
        #Placeholders
        # dim N * 25 * 1000 * 1
        X  = tf.placeholder(dtype=tf.float32, shape=(None, 25, 1000, 1), name='X')
        Y  = tf.placeholder(dtype=tf.float32, shape=None, name='Y')
        T  = tf.placeholder(tf.bool)
        LR = tf.placeholder(tf.float32)

        H = tf.layers.conv2d(X, filters=40, kernel_size=(1,25), strides=(1, 1), padding='valid', activation=None , kernel_initializer=initK)
        H = tf.layers.batch_normalization(H, momentum=0.1, training=T) 
        # dim N * 25 * 976 * 40

        H = tf.layers.conv2d(H,filters=40, kernel_size=(25,1), strides=(1, 1), padding='valid', activation=None, kernel_initializer=initK)
        H = tf.layers.batch_normalization(H, momentum=0.1, training=T) 
        # dim N * 1 * 976 *40

        H = tf.square(H)
        H = tf.layers.average_pooling2d(H , pool_size=(1,75) , strides=(1,15), padding='valid')
        # dim N * 1 * 61 * 40
        
        H = tf.log(H)
        H = tf.layers.dropout(H, rate=0.6, training=T)

        H = tf.layers.conv2d(H, filters=4, kernel_size=(1,61), padding='valid', activation=None , kernel_initializer=initK)   
        # dim (N)*1*1*4

        Y_pred = tf.reshape(H,[-1,4], name='Y_pred')
        # dim (N)*4

        Y_pred = tf.nn.softmax(Y_pred, name='Y_pred')  # w w/o result simular
        #Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))
        Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
        Update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(Update_ops):
            Train_step = tf.train.AdamOptimizer(LR).minimize(Loss)

        init2 = tf.global_variables_initializer()
        
        
        ### training model ###
        batch_size = train_len / 20
        lr=0.005

        loss_hist   = []
        loss_hist2  = []
        train_hist  = []
        test_hist2  = []

        with tf.Session(graph=gf) as se:
            se.run(init2)
            for epoch in range(60):
                id_list = get_batch_id(batch_size, train_len)
                for batch_id in id_list:
                    batch_x = x_train[batch_id]
                    batch_y = y_train[batch_id]

                    print(batch_x.shape)
                    print(batch_y.shape) 
                    
                    _ , loss_i =  se.run([Train_step,Loss] , feed_dict={X: batch_x, Y: batch_y, T: True, LR:lr})

                    print(loss_i)

                loss_test = se.run(Loss,feed_dict={X: x_test , Y: y_test , T: False})

                y_test_pred      = se.run(Y_pred, feed_dict={X: x_test, T: False})
                Y_correct_test   = tf.equal(tf.argmax(y_test_pred,1), tf.argmax(y_test,1))
                Y_correct_test   = tf.cast(Y_correct_test, tf.float32)
                acc_tf_test      = tf.reduce_mean(Y_correct_test)             
                acc_test2        = se.run(acc_tf_test)   

                train_loss = 0.0
                train_acc = 0.0

                for batch_id in id_list:
                    batch_x = x_train[batch_id]
                    batch_y = y_train[batch_id]

                    loss_batch_train = se.run(Loss,feed_dict={X: batch_x , Y: batch_y , T: False})
                    Y_train_pred     = se.run(Y_pred, feed_dict={X: batch_x, Y: batch_y, T: False})
                    Y_correct_train  = tf.equal(tf.argmax(Y_train_pred,1), tf.argmax(batch_y,1))
                    Y_correct_train  = tf.cast(Y_correct_train, tf.float32)
                    acc_tf_train     = tf.reduce_mean(Y_correct_train)             
                    acc_train        = se.run(acc_tf_train)
                    train_loss = train_loss + loss_batch_train * batch_x.shape[0] / x_train.shape[0]
                    train_acc = train_acc + acc_train * batch_x.shape[0] / x_train.shape[0]

                loss_hist.append(train_loss)
                loss_hist2.append(loss_test)
                train_hist.append(train_acc)
                test_hist2.append(acc_test2)

                print('Epoch:', epoch, '| test: %.4f' % acc_test2, '| train: %.4f' % acc_train,
                      '| train Loss: %.4f' % loss_i, '| test Loss: %.4f' % loss_test)

        plt.figure()
        plt.plot(train_hist,color='red',label='train')
        plt.plot(test_hist2,color='green',label='test')
        
        plt.ylabel('Test_acc , Train_acc')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(loss_hist, color='red',label='train loss')
        plt.plot(loss_hist2,color='green',label='test  loss')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()