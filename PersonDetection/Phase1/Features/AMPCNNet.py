"""
Created by Bing Liu
Human detection CNN model of amplitude
"""

# Importing Python libraries
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import Header as hd
 
class CNNNet:
    channel = 1
    n_classes = 2
    x = tf.placeholder(tf.float64, (None, 0, 0, channel))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float64)       # For fully-connected layers
    keep_prob_conv = tf.placeholder(tf.float64)  # For convolutional layers
    
    def __init__(self, n_out=2, mu=0, sigma=0.1, learning_rate=0.001):
        # Hyperparameters
        self.mu = mu
        self.sigma = sigma

        # Layer 1 (Convolutional)
        self.conv1_W = tf.Variable(tf.truncated_normal(shape=(1, 4, self.channel, 8), mean = self.mu, stddev = self.sigma))
        self.conv1_b = tf.Variable(tf.zeros(8))
        self.conv1   = tf.nn.conv2d(self.x, self.conv1_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv1_b

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
        
    def y_predict(self, X_data, BATCH_SIZE=16):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.float32)
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x = X_data[offset:offset+BATCH_SIZE]
            y_pred[offset:offset+BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1), 
                               feed_dict={self.x:batch_x, self.keep_prob:1, self.keep_prob_conv:1})
        return y_pred
    
    def evaluate(self, X_data, y_data, Valid_Writer, Valid_Writer_Index, BATCH_SIZE=16, ):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            summary, accuracy = sess.run([self.merged,  self.accuracy_operation], 
                                feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.0, self.keep_prob_conv: 1.0 })
            Valid_Writer.add_summary(summary, Valid_Writer_Index)
            Valid_Writer_Index = Valid_Writer_Index +1
            total_accuracy += (accuracy * len(batch_x))
        return Valid_Writer_Index, total_accuracy / num_examples