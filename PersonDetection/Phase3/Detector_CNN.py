"""
Created by Bing Liu
Build CNN to detect person in phase 3
"""
# Importing Python libraries
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import Common as com
import os
import warnings
warnings.filterwarnings("ignore")
class CNNNet:
    channel = 1
    n_classes = 2
    x = tf.placeholder(tf.float32, (None, 0, 0, channel))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float32)       # For fully-connected layers
    keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers
    
    def __init__(self, n_out=10, mu=0, sigma=0.1, learning_rate=0.0001): #learning_rate=0.001
        # Hyperparameters
        self.mu = mu
        self.sigma = sigma

        # Layer 1 (Convolutional)
        self.conv1_W = tf.Variable(tf.truncated_normal(shape=(1, 4, self.channel, 8), mean = self.mu, stddev = self.sigma))
        self.conv1_b = tf.Variable(tf.zeros(8))
        self.conv1   = tf.nn.conv2d(self.x, self.conv1_W, strides=[1, 2, 1, 1], padding='SAME') + self.conv1_b

        # ReLu Activation.
        self.conv1 = tf.nn.relu(self.conv1)

        # Layer 2 (Convolutional)
        self.conv2_W = tf.Variable(tf.truncated_normal(shape=(1, 4, 8, 8), mean = self.mu, stddev = self.sigma))
        self.conv2_b = tf.Variable(tf.zeros(8))
        self.conv2   = tf.nn.conv2d(self.conv1, self.conv2_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b

        # ReLu Activation.
        self.conv2 = tf.nn.relu(self.conv2)

        # Layer 3 (Convolutional)
        self.conv3_W = tf.Variable(tf.truncated_normal(shape=(1, 4, 8, 8), mean = self.mu, stddev = self.sigma))
        self.conv3_b = tf.Variable(tf.zeros(8))
        self.conv3   = tf.nn.conv2d(self.conv2, self.conv3_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b

        # ReLu Activation.
        self.conv3 = tf.nn.relu(self.conv3)

        # Layer 4 (Convolutional)
        self.conv4_W = tf.Variable(tf.truncated_normal(shape=(1, 4, 8, 8), mean = self.mu, stddev = self.sigma))
        self.conv4_b = tf.Variable(tf.zeros(8))
        self.conv4   = tf.nn.conv2d(self.conv3, self.conv4_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b

        # ReLu Activation.
        self.conv4 = tf.nn.relu(self.conv4)

        # Flatten.
        self.fc0   = flatten(self.conv4)

        # Layer 5 (Fully Connected)
        shape_X = np.shape(self.fc0)[1].value
        self.fc1_W = tf.Variable(tf.truncated_normal(shape=(shape_X, 2), mean = self.mu, stddev = self.sigma))
        self.fc1_b = tf.Variable(tf.zeros(2))
        self.fc1   = tf.matmul(self.fc0, self.fc1_W) + self.fc1_b

        # ReLu Activation.
        self.fc1    = tf.nn.relu(self.fc1)
        self.fc1    = tf.nn.dropout(self.fc1, self.keep_prob) # dropout

        self.logits = tf.matmul(self.fc0, self.fc1_W) + self.fc1_b

        # Training operation
        self.one_hot_y = tf.one_hot(self.y, n_out)
        #self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = self.one_hot_y, logits = self.logits)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.one_hot_y, logits = self.logits)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        tf.summary.scalar('Loss', self.loss_operation)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)
        

        # Accuracy operation
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', self.accuracy_operation)
        
        # Saving all variables
        self.merged = tf.summary.merge_all() 
        self.saver = tf.train.Saver()
                
    def y_predict(self, X_data, BATCH_SIZE=4):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.float32)
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            if offset<num_examples-BATCH_SIZE:
                batch_x = X_data[offset:offset+BATCH_SIZE]
                batch_x = np.reshape(batch_x,(BATCH_SIZE,np.shape(X_data)[1], 1,1))
                y_pred[offset:offset+BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1), 
                                feed_dict={self.x:batch_x, self.keep_prob:1, self.keep_prob_conv:1})
        return y_pred
    
    def evaluate(self, X_data, y_data, Valid_Writer, Valid_Writer_Index, BATCH_SIZE=4, ):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            if offset+BATCH_SIZE<num_examples:
                batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
                batch_x = np.reshape(batch_x, (BATCH_SIZE, batch_x.shape[1],1,1 ))
                summary, accuracy = sess.run([self.merged,  self.accuracy_operation], 
                                    feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.0, self.keep_prob_conv: 1.0 })
                Valid_Writer.add_summary(summary, Valid_Writer_Index)
                Valid_Writer_Index = Valid_Writer_Index +1
                total_accuracy += (accuracy * len(batch_x))
        return Valid_Writer_Index, total_accuracy / num_examples

def Train(EPOCHS,
        BATCH_SIZE,
        CNN_Folder,
        Freqs_Num,x_train, y_train, x_test, y_test ):
    if tf.gfile.Exists(CNN_Folder):
        tf.gfile.DeleteRecursively(CNN_Folder)
    tf.gfile.MakeDirs(CNN_Folder)

    log_dir = CNN_Folder + '/log'
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    
    CNNNet.x = tf.placeholder(np.float, (None, Freqs_Num, 1, 1))
    CNN_Model = CNNNet(n_out = CNNNet.n_classes)

    Train_Writer_Index = 0
    Valid_Writer_Index = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Train: {} Tes: {}".format(len(x_train), len(x_test)))

        Train_Writer = tf.summary.FileWriter(log_dir + '/Training', sess.graph)
        Valid_Writer = tf.summary.FileWriter(log_dir + '/Validation', sess.graph)
        
        for i in range(EPOCHS):
            s = np.arange(len(x_train))
            np.random.shuffle(s)
            x_train = x_train[s]
            y_train = y_train[s]

            num_examples = len(y_train)

            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                if end>num_examples:
                    break
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                batch_x = np.reshape(batch_x, (BATCH_SIZE,Freqs_Num,1,1))
                summary, _ = sess.run([CNN_Model.merged, CNN_Model.training_operation],
                                        feed_dict={CNNNet.x: batch_x, CNNNet.y: batch_y, CNNNet.keep_prob : 0.5, CNNNet.keep_prob_conv: 0.7})
                Train_Writer.add_summary(summary, Train_Writer_Index)
                Train_Writer_Index = Train_Writer_Index + 1

            Valid_Writer_Index, validation_accuracy = CNN_Model.evaluate(x_test, y_test, Valid_Writer, Valid_Writer_Index)
            print("EPOCH :{} Validation Accuracy = {:.3f}%".format(i+1, validation_accuracy*100))

            if validation_accuracy*100> 95:
                break

        CNN_Model.saver.save(sess, os.path.join(CNN_Folder, 'CNN'))
        msg = "Model saved: " + os.path.join(CNN_Folder, 'CNN')
        print(msg)


def Test(x_test, y_test, CNN_Folder, Freqs_Num, Model_Name='CNN'):
    warnings.filterwarnings("ignore")

    x_test = np.asarray(x_test, dtype=np.float)
    y_test = np.asarray(y_test, dtype=np.float32)
    n_test = x_test.shape[0]

    CNNNet.x = tf.placeholder(np.float, (None, Freqs_Num, 1, 1))
    CNN_Model = CNNNet(n_out = 2)

    msg = "Number of testing examples: {} ".format(n_test)
    print(msg)
    with tf.Session() as sess:
        msg = "Loading model from " + os.path.join(CNN_Folder, Model_Name)
        print(msg)
        CNN_Model.saver.restore(sess, os.path.join(CNN_Folder, Model_Name))
        y_pred = CNN_Model.y_predict(x_test)
        test_accuracy = sum(y_test == y_pred)/len(y_test)
        msg = "Test Accuracy = {:.1f}%".format(test_accuracy*100)
        print(msg)

        predicted  = np.array(y_pred).astype("int")
        actual = np.array(y_test).astype("int")

        TP = np.count_nonzero(predicted * actual)
        TN = np.count_nonzero((predicted - 1) * (actual - 1))
        FP = np.count_nonzero(predicted * (actual - 1))
        FN = np.count_nonzero((predicted - 1) * actual)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        msg_arr = []
        #msg_arr.append("TP = {:d}".format(TP))
        #msg_arr.append("TN = {:d}".format(TN))
        #msg_arr.append("FP = {:d}".format(FP))
        #msg_arr.append("FN = {:d}".format(FN))
        #msg_arr.append("Precision = {:.1f}%".format(precision*100))
        #msg_arr.append("Recall = {:.1f}%".format(recall*100))
        #msg_arr.append("F1 = {:.1f}%".format(f1*100))

        msg_arr.append("{:d}".format(TP))
        msg_arr.append("{:d}".format(TN))
        msg_arr.append("{:d}".format(FP))
        msg_arr.append("{:d}".format(FN))
        msg_arr.append("{:.1f}%".format(precision*100))
        msg_arr.append("{:.1f}%".format(recall*100))
        msg_arr.append("{:.1f}%".format(f1*100))
    return TP, TN, FP, FN, test_accuracy*100
