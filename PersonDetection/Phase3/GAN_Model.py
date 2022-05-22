"""
Created by Bing Liu
Build GAN model
"""
from __future__ import division
from __future__ import print_function
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from CNN_Base import *
from Com_GAN import *
import RFDataSetRawActive
import RFDataSetPowerAvg
import Com_Power
import csv

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))
  
def gen_random(mode, size):
    if mode=='normal01': return np.random.normal(0,1,size=size)
    if mode=='uniform_signed': return np.random.uniform(-1,1,size=size)
    if mode=='uniform_unsigned': return np.random.uniform(0,1,size=size)

class DCGAN(object):
  def __init__( self, 
                sess,
                gene_sample_num,
                input_height,
                input_width,
                batch_size,
                learn_rate_D,
                learn_rate_G,
                output_height,
                output_width,
                y_dim,
                z_dim=100,
                K_D_H0=3,
                K_D_W0=3,
                K_D_H1=3,
                K_D_W1=3,
                K_D_H2=5,
                K_D_W2=5,
                K_D_H3=5,
                K_D_W3=5,
                S_D_H=2,
                S_D_W=2,
                K_G_H0=5,
                K_G_W0=5,
                K_G_H1=5,
                K_G_W1=5,
                K_G_H2=5,
                K_G_W2=5,
                K_G_H3=5,
                K_G_W3=5,
                S_G_H=2,
                S_G_W=2,
                gf_dim=64,
                df_dim=64,
                gfc_dim=1024,
                dfc_dim=1024,
                dataset=None,
                max_to_keep=1,
                checkpoint_dir='ckpts',
                loss_dir='lossdir',
                sample_dir='samples',
                generated_data_dir='',
                train_data_dir='',
                figure_dir = "figures",
                out_dir='./out',
                data_dir='./data',
                freq_num = 784,
                raw_sample_num = 1200,
                train_location = '01',
                test_location = '05'):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim:   Dimension of dim for y. [None]
      z_dim:   Dimension of dim for Z. [100]
      gf_dim:  Dimension of gen filters in first conv layer. [64]
      df_dim:  Dimension of discrim filters in first conv layer. [64]
      gfc_dim: Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: Dimension of discrim units for fully connected layer. [1024]
    """
    self.conditioning = False
    self.input_random = False
    self.sess = sess
    self.gene_sample_num = gene_sample_num
    self.batch_size = batch_size
    self.learn_rate_D = learn_rate_D
    self.learn_rate_G = learn_rate_G
    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width
    self.y_dim = y_dim
    self.z_dim = z_dim
    self.K_D_H0 = K_D_H0
    self.K_D_W0 = K_D_W0
    self.K_D_H1 = K_D_H1
    self.K_D_W1 = K_D_W1
    self.K_D_H2 = K_D_H2
    self.K_D_W2 = K_D_W2
    self.K_D_H3 = K_D_H3
    self.K_D_W3 = K_D_W3
    self.S_D_H = S_D_H
    self.S_D_W = S_D_W
    self.K_G_H0 = K_G_H0
    self.K_G_W0 = K_G_W0
    self.K_G_H1 = K_G_H1
    self.K_G_W1 = K_G_W1
    self.K_G_H2 = K_G_H2
    self.K_G_W2 = K_G_W2
    self.K_G_H3 = K_G_H3
    self.K_G_W3 = K_G_W3
    self.S_G_H = S_G_H
    self.S_G_W = S_G_W

    self.gf_dim = gf_dim
    self.df_dim = df_dim
    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn0 = batch_norm(name='d_bn0')
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')
    self.d_bn4 = batch_norm(name='d_bn4')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')
    self.g_bn4 = batch_norm(name='g_bn4')

    self.checkpoint_dir = checkpoint_dir
    self.loss_dir = loss_dir
    self.train_data_dir = train_data_dir
    self.data_dir = data_dir
    self.out_dir = out_dir
    self.figure_dir = figure_dir
    self.max_to_keep = max_to_keep
    self.dataset = dataset
    self.freq_num = freq_num
    self.raw_sample_num = raw_sample_num
    self.freqs_list_train = []
    self.train_location = train_location
    self.test_location = test_location
    
    if self.dataset == Dataset.MNIST or self.dataset == Dataset.MNIST_1D:
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]
    elif self.dataset == Dataset.Raw:
      self.freqs_list_train, self.data_X, self.data_y = self.load_Raw()
      self.c_dim = self.data_X[0].shape[-1]
    else:
      #self.freqs_list_train, self.data_X, self.data_y = self.load_Power_Avg()
      self.freqs_list_train, self.data_X_Yes, self.data_y_Yes, self.data_X_No, self.data_y_No = self.load_Power_Avg(self.train_location)
      self.c_dim = self.data_X_Yes[0].shape[-1]
    
    self.build_model()

    csv_file_name = os.path.join(self.loss_dir, "loss.txt")
    self.loss_D = []
    self.loss_G = []
    if os.path.exists(csv_file_name):
      with open(csv_file_name, 'r') as f:
        self.loss_D, self.loss_G = get_loss(f)
    self.csv_file = open(csv_file_name, 'a+')
    self.loss_file_writer = csv.writer(self.csv_file)
    self.loss_figure_file_name = os.path.join(self.loss_dir, "loss.png")
    
  def build_model(self):
    self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    input_dims = [self.input_height, self.input_width, self.c_dim]
    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + input_dims, name='real_samples')
    inputs = self.inputs

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z, self.y)
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
    self.sampler            = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

    self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

  def train(self, config):

    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

    tf.global_variables_initializer().run()

    if config.G_img_sum:
      self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    else:
      self.g_sum = merge_summary([self.z_sum, self.d__sum, self.d_loss_fake_sum, self.g_loss_sum])

    self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

    self.writer = SummaryWriter(os.path.join(self.out_dir, "logs"), self.sess.graph)

    #sample_z = gen_random(config.z_dist, size=(self.batch_size , self.z_dim))
   
    if self.dataset == Dataset.Power_Avg:
      sample_inputs = self.data_X_No[0:self.batch_size]
      sample_labels = self.data_y_No[0:self.batch_size]
    else:
      sample_inputs = self.data_X[0:self.batch_size]
      sample_labels = self.data_y[0:self.batch_size]

  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load_model(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
        
        if self.dataset == Dataset.Power_Avg:
          train_sample_num = len(self.data_X_No)
        else:
          train_sample_num = len(self.data_X)

        batch_idxs = int(train_sample_num/config.batch_size)
        
        for idx in xrange(0, batch_idxs):
          start_index = idx*config.batch_size
          end_index = min((idx+1)*config.batch_size, train_sample_num)
          #batch_inputs = self.data_X[start_index:end_index]
          #batch_labels = self.data_y[start_index:end_index]

          if self.dataset == Dataset.Power_Avg:
            batch_inputs = self.data_X_Yes[start_index:end_index]
            batch_labels = self.data_y_Yes[start_index:end_index]
          else:
            batch_inputs = self.data_X[start_index:end_index]
            batch_labels = self.data_y[start_index:end_index]

          #batch_z = gen_random(config.z_dist, size=[config.batch_size, self.z_dim]).astype(np.float32)

        if self.dataset == Dataset.Power_Avg:
          batch_z = np.reshape(self.data_X_No[start_index:end_index], (config.batch_size, self.z_dim))
        else:
          print(np.shape(self.data_X[start_index:end_index]))
          print((config.batch_size, self.z_dim))
          batch_z = np.reshape(self.data_X[start_index:end_index], (config.batch_size, self.z_dim))

          batch_z = batch_z + gen_random(config.z_dist, size=[config.batch_size, self.z_dim]).astype(np.float32) /100.0
          # Update D network
          #_, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={ self.inputs: batch_inputs, self.z: batch_z, self.y:batch_labels,})
          #_, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={ self.inputs: batch_inputs, batch_inputs, self.y:batch_labels,})
          _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={ self.inputs: batch_inputs, self.z: batch_z, self.y:batch_labels,})
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.z: batch_z, self.y:batch_labels,})
          #_, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.z: np.reshape(batch_inputs, (config.batch_size, (28,28)))), self.y:batch_labels,})
          self.writer.add_summary(summary_str, counter)

          #_, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={ self.z: batch_z, self.y:batch_labels })
          #self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y:batch_labels})
          errD_real = self.d_loss_real.eval({self.inputs: batch_inputs, self.y:batch_labels})
          errG = self.g_loss.eval({self.z: batch_z, self.y: batch_labels})

          loss_D = errD_fake+errD_real
          loss_G = errG
          out_string = "[%8d Epoch:[%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.4f, g_loss: %.4f" \
            % (counter, epoch, config.epoch, idx, batch_idxs, time.time() - start_time, loss_D, loss_G)
          print(out_string)

          csv_out = ['%8d Epoch:%2d/%2d %4d/%4d'%(counter, epoch, config.epoch, idx, batch_idxs),
                                           'd_loss:',
                                           '%.4f'%(loss_D),
                                           'g_loss:',
                                            '%.4f'%(loss_G)]


          self.loss_D.append(loss_D)
          self.loss_G.append(loss_G)
          self.loss_file_writer.writerow(csv_out)

          if np.mod(counter, config.sample_freq) == 0:

            draw_loss(self.loss_figure_file_name, self.loss_D, self.loss_G)

            if self.dataset == Dataset.Power_Avg:
              start = np.random.randint(0,int(len(self.data_X_No)/4), 1)[0]
            else:
              start = np.random.randint(0,int(len(self.data_X)/4), 1)[0]

            end = start + config.batch_size

            if self.dataset == Dataset.Power_Avg:
              sample_z = np.reshape(self.data_X_No[start:end], (config.batch_size, self.z_dim))
            else:
              sample_z = np.reshape(self.data_X[start:end], (config.batch_size, self.z_dim))


            samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
                                                    feed_dict={ self.z: sample_z,
                                                                self.inputs: sample_inputs,
                                                                self.y:sample_labels,})
            if self.dataset == Dataset.MNIST:
              save_images(samples, image_manifold_size(samples.shape[0]),
                    '{}/train_{:08d}.png'.format(config.sample_dir, counter))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            elif self.dataset == Dataset.MNIST_1D:
              samples = np.reshape(samples, (self.batch_size, 28,28,1))
              save_images(samples, image_manifold_size(samples.shape[0]),
                    '{}/train_{:08d}.png'.format(config.sample_dir, counter))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            elif self.dataset == Dataset.Raw:
              power_avg_arry = Com_Power.Caculate_Freqs_Powers_Arr(batch_inputs*255)
              figure_file_name = '{}/train_{:08d}_R.png'.format(config.sample_dir, counter)
              Com_Power.Draw_Freqs_Figure_Single_Sub_Plot(figure_file_name, self.freqs_list_train, power_avg_arry, 4, 2)

              power_avg_arry = Com_Power.Caculate_Freqs_Powers_Arr(samples*255)
              figure_file_name = '{}/train_{:08d}_G.png'.format(config.sample_dir, counter)
              Com_Power.Draw_Freqs_Figure_Single_Sub_Plot(figure_file_name, self.freqs_list_train, power_avg_arry, 4, 2)
            else:
              samples = -samples*40.0
              
              if self.batch_size > 4 :
                plot_size_col = int(np.sqrt(self.batch_size))
              else:
                plot_size_col = 1
              plot_size_row = int(self.batch_size/plot_size_col)
              figure_file_name = '{}/train_{:08d}_real.png'.format(config.sample_dir, counter)
              Com_Power.Draw_Freqs_Figure_Single_Sub_Plot_Label(figure_file_name, self.freqs_list_train, -batch_inputs.reshape((self.batch_size,-1))*40, batch_labels, plot_size_row, plot_size_col)
              #Com_Power.Draw_Freqs_Figure_Single_Sub_Plot(figure_file_name, self.freqs_list_train, batch_inputs.reshape((self.batch_size,-1)), plot_size_row, plot_size_col)

              figure_file_name = '{}/train_{:08d}_gene.png'.format(config.sample_dir, counter)
              Com_Power.Draw_Freqs_Figure_Single_Sub_Plot(figure_file_name, self.freqs_list_train, samples, plot_size_row, plot_size_col)
              
              '''
              figure_file_name = '{}/train_{:08d}_gene_avg.png'.format(config.sample_dir, counter)
              samples = np.average(samples,axis=0)
              samples = samples.flatten()
              Com_Power.Draw_Freqs_Figure_One(figure_file_name, self.freqs_list_train, samples)
              '''

          if np.mod(counter, config.ckpt_freq) == 0:
            self.save(config.checkpoint_dir, counter)
          
          counter += 1

          if counter == 29230:
            return

  def discriminator(self, input, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      if self.conditioning:
        x = conv_cond_concat(input, yb)

      if self.conditioning:
        cnn_input = x
      else:
        cnn_input = input
      h0 = lrelu(conv2d(cnn_input,
                        self.c_dim + self.y_dim, 
                        k_h=self.K_D_H0, 
                        k_w=self.K_D_W0, 
                        d_h=self.S_D_H, 
                        d_w=self.S_D_W, 
                        stddev = self.learn_rate_D,
                        name='d_h0_conv'))
      if self.conditioning:
        h0 = conv_cond_concat(h0, yb)

      h1 = lrelu(conv2d(h0,
                        self.c_dim + self.y_dim, 
                        k_h=self.K_D_H1, 
                        k_w=self.K_D_W1, 
                        d_h=self.S_D_H, 
                        d_w=self.S_D_W,
                        stddev = self.learn_rate_D,
                        name='d_h1_conv'))
      if self.conditioning:
        h1 = conv_cond_concat(h1, yb)

      h2 = lrelu(self.d_bn1(conv2d( h1, 
                                    self.df_dim + self.y_dim, 
                                    k_h=self.K_D_H2, 
                                    k_w=self.K_D_W2, 
                                    d_h=self.S_D_H, 
                                    d_w=self.S_D_W, 
                                    stddev = self.learn_rate_D,
                                    name='d_h2_conv')))
      h2 = tf.reshape(h2, [self.batch_size, -1])

      if self.conditioning:
        h2 = concat([h2, y], 1)
      
      h3 = lrelu(self.d_bn2(linear(h2, int(self.dfc_dim), 'd_h3_lin')))
      if self.conditioning:
        h3 = concat([h3, y], 1)
      '''
      h4 = lrelu(self.d_bn3(linear(h3, int(self.dfc_dim/2), 'd_h4_lin')))
      if self.conditioning:
        h4 = concat([h4, y], 1)

      h5 = lrelu(self.d_bn4(linear(h4, int(self.dfc_dim/4), 'd_h5_lin')))
      if self.conditioning:
        h5 = concat([h5, y], 1)
      '''
      h6 = linear(h3, 1, 'd_h6_lin')
      
      return tf.nn.sigmoid(h6), h6

  def generator(self, z, y=None):

    with tf.variable_scope("generator") as scope:
      s_h, s_w = self.output_height, self.output_width

      if s_h==1:
        s_h0, s_h1, s_h2 = max(int(s_h/2),1), max(int(s_h/4),1), max(int(s_h/8),1)
        s_w0, s_w1, s_w2 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8),1)
      else:
        #s_h0, s_h1, s_h2 = max(int(s_h/2),1), max(int(s_h/4),1), max(int(s_h/8)+1,1) #power
        #s_w0, s_w1, s_w2 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8)+1,1) #power
        s_h0, s_h1, s_h2 = max(int(s_h/2)+1,1), max(int(s_h/4)+1,2), max(int(s_h/8)+1,2)  #raw
        s_w0, s_w1, s_w2 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8),1)  #raw

      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      if self.conditioning:
        z = concat([z, y], 1)

      h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
      if self.conditioning:
        h0 = concat([h0, y], 1)

      h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h2*s_w2, 'g_h1_0_lin')))
      h1 = tf.reshape(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2])
      if self.conditioning:
        h1 = conv_cond_concat(h1, yb)

      h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h1, s_w1, self.gf_dim * 2],
                                          k_h=self.K_G_H0, 
                                          k_w=self.K_G_W0, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h1_con')))
      if self.conditioning:
        h2 = conv_cond_concat(h2, yb)

      h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, s_h0, s_w0, self.gf_dim * 2],
                                          k_h=self.K_G_H1, 
                                          k_w=self.K_G_W1, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h2_con')))
      if self.conditioning:
        h3 = conv_cond_concat(h3, yb)

      h4 = tf.nn.sigmoid(deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim],
                                    k_h=self.K_G_H3, 
                                    k_w=self.K_G_W3, 
                                    d_h=self.S_G_H, 
                                    d_w=self.S_G_W, 
                                    stddev = self.learn_rate_G,
                                    name='g_h3_con'))

      return h4

  def sampler(self, z, y=None):

    with tf.variable_scope("generator") as scope: 
      scope.reuse_variables()
      s_h, s_w = self.output_height, self.output_width
      if s_h==1:
        s_h0, s_h1, s_h2 = max(int(s_h/2),1), max(int(s_h/4),1), max(int(s_h/8),1)
        s_w0, s_w1, s_w2 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8),1)
      else:
        #s_h0, s_h1, s_h2 = max(int(s_h/2),1), max(int(s_h/4),1), max(int(s_h/8)+1,1) #power
        #s_w0, s_w1, s_w2 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8)+1,1) #power
        s_h0, s_h1, s_h2 = max(int(s_h/2)+1,1), max(int(s_h/4)+1,2), max(int(s_h/8)+1,2)  #raw
        s_w0, s_w1, s_w2 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8),1)  #raw

      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      if self.conditioning:
        z = concat([z, y], 1)

      h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
      if self.conditioning:
        h0 = concat([h0, y], 1)

      h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h2*s_w2, 'g_h1_0_lin')))
      h1 = tf.reshape(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2])
      if self.conditioning:
        h1 = conv_cond_concat(h1, yb)

      h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h1, s_w1, self.gf_dim * 2],
                                          k_h=self.K_G_H0, 
                                          k_w=self.K_G_W0, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h1_con')))
      if self.conditioning:
        h2 = conv_cond_concat(h2, yb)

      h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, s_h0, s_w0, self.gf_dim * 2],
                                          k_h=self.K_G_H1, 
                                          k_w=self.K_G_W1, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h2_con')))
      if self.conditioning:
        h3 = conv_cond_concat(h3, yb)

      h4 = tf.nn.sigmoid(deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim],
                                    k_h=self.K_G_H3, 
                                    k_w=self.K_G_W3, 
                                    d_h=self.S_G_H, 
                                    d_w=self.S_G_W, 
                                    stddev = self.learn_rate_G,
                                    name='g_h3_con'))

      return h4

  def sampler_5(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      s_h, s_w = self.output_height, self.output_width

      if s_h==1:
        s_h0, s_h1, s_h2, s_h3 = max(int(s_h/2),1), max(int(s_h/4),1), max(int(s_h/8),1), max(int(s_h/16),1)
        s_w0, s_w1, s_w2, s_w3 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8),1), max(int(s_w/16),1)
      else:
        s_h0, s_h1, s_h2, s_h3 = max(int(s_h/2),1), max(int(s_h/4),1), max(int(s_h/8),1), max(int(s_h/16),1)
        s_w0, s_w1, s_w2, s_w3 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8),1), max(int(s_w/16),1)

      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      if self.conditioning:
        z = concat([z, y], 1)

      h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
      if self.conditioning:
        h0 = concat([h0, y], 1)

      h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h3*s_w3, 'g_h1_0_lin')))
      h1 = tf.reshape(h1, [self.batch_size, s_h3, s_w3, self.gf_dim * 2])
      if self.conditioning:
        h1 = conv_cond_concat(h1, yb)

      h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2],
                                          k_h=self.K_G_H0, 
                                          k_w=self.K_G_W0, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h2_con')))
      if self.conditioning:
        h2 = conv_cond_concat(h2, yb)

      h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, s_h1, s_w1, self.gf_dim * 2],
                                          k_h=self.K_G_H1, 
                                          k_w=self.K_G_W1, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h3_con')))
      if self.conditioning:
        h3 = conv_cond_concat(h3, yb)
      h4 = tf.nn.relu(self.g_bn4(deconv2d(h3, [self.batch_size, s_h0, s_w0, self.gf_dim * 2],
                                          k_h=self.K_G_H1, 
                                          k_w=self.K_G_W1, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h4_con')))
      if self.conditioning:
        h4 = conv_cond_concat(h4, yb)

      h5 = tf.nn.sigmoid(deconv2d(h4, [self.batch_size, s_h, s_w, self.c_dim],
                                    k_h=self.K_G_H3, 
                                    k_w=self.K_G_W3, 
                                    d_h=self.S_G_H, 
                                    d_w=self.S_G_W, 
                                    stddev = self.learn_rate_G,
                                    name='g_h5_con'))

      return h5

  def generator_5(self, z, y=None):

    with tf.variable_scope("generator") as scope:
      s_h, s_w = self.output_height, self.output_width

      if s_h==1:
        s_h0, s_h1, s_h2, s_h3 = max(int(s_h/2),1), max(int(s_h/4),1), max(int(s_h/8),1), max(int(s_h/16),1)
        s_w0, s_w1, s_w2, s_w3 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8),1), max(int(s_w/16),1)
      else:
        s_h0, s_h1, s_h2, s_h3 = max(int(s_h/2),1), max(int(s_h/4),1), max(int(s_h/8),1), max(int(s_h/16),1)
        s_w0, s_w1, s_w2, s_w3 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8),1), max(int(s_w/16),1)

      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      if self.conditioning:
        z = concat([z, y], 1)

      h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
      if self.conditioning:
        h0 = concat([h0, y], 1)

      h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h3*s_w3, 'g_h1_0_lin')))
      h1 = tf.reshape(h1, [self.batch_size, s_h3, s_w3, self.gf_dim * 2])
      if self.conditioning:
        h1 = conv_cond_concat(h1, yb)

      h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2],
                                          k_h=self.K_G_H0, 
                                          k_w=self.K_G_W0, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h2_con')))
      if self.conditioning:
        h2 = conv_cond_concat(h2, yb)

      h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, s_h1, s_w1, self.gf_dim * 2],
                                          k_h=self.K_G_H1, 
                                          k_w=self.K_G_W1, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h3_con')))
      if self.conditioning:
        h3 = conv_cond_concat(h3, yb)

      h4 = tf.nn.relu(self.g_bn4(deconv2d(h3, [self.batch_size, s_h0, s_w0, self.gf_dim * 2],
                                          k_h=self.K_G_H1, 
                                          k_w=self.K_G_W1, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h4_con')))
      if self.conditioning:
        h4 = conv_cond_concat(h4, yb)
      #h4 = tf.layers.max_pooling2d(h4, 1, 2)

      h5 = tf.nn.sigmoid(deconv2d(h4, [self.batch_size, s_h, s_w, self.c_dim],
                                    k_h=self.K_G_H3, 
                                    k_w=self.K_G_W3, 
                                    d_h=self.S_G_H, 
                                    d_w=self.S_G_W, 
                                    stddev = self.learn_rate_G,
                                    name='g_h5_con'))

      return h5

  def sampler_5(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      s_h, s_w = self.output_height, self.output_width

      if s_h==1:
        s_h0, s_h1, s_h2, s_h3 = max(int(s_h/2),1), max(int(s_h/4),1), max(int(s_h/8),1), max(int(s_h/16),1)
        s_w0, s_w1, s_w2, s_w3 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8),1), max(int(s_w/16),1)
      else:
        s_h0, s_h1, s_h2, s_h3 = max(int(s_h/2),1), max(int(s_h/4),1), max(int(s_h/8),1), max(int(s_h/16),1)
        s_w0, s_w1, s_w2, s_w3 = max(int(s_w/2),1), max(int(s_w/4),1), max(int(s_w/8),1), max(int(s_w/16),1)

      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      if self.conditioning:
        z = concat([z, y], 1)

      h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
      if self.conditioning:
        h0 = concat([h0, y], 1)

      h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h3*s_w3, 'g_h1_0_lin')))
      h1 = tf.reshape(h1, [self.batch_size, s_h3, s_w3, self.gf_dim * 2])
      if self.conditioning:
        h1 = conv_cond_concat(h1, yb)

      h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2],
                                          k_h=self.K_G_H0, 
                                          k_w=self.K_G_W0, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h2_con')))
      if self.conditioning:
        h2 = conv_cond_concat(h2, yb)

      h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, s_h1, s_w1, self.gf_dim * 2],
                                          k_h=self.K_G_H1, 
                                          k_w=self.K_G_W1, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h3_con')))
      if self.conditioning:
        h3 = conv_cond_concat(h3, yb)
      h4 = tf.nn.relu(self.g_bn4(deconv2d(h3, [self.batch_size, s_h0, s_w0, self.gf_dim * 2],
                                          k_h=self.K_G_H1, 
                                          k_w=self.K_G_W1, 
                                          d_h=self.S_G_H, 
                                          d_w=self.S_G_W, 
                                          stddev = self.learn_rate_G,
                                          name='g_h4_con')))
      if self.conditioning:
        h4 = conv_cond_concat(h4, yb)

      h5 = tf.nn.sigmoid(deconv2d(h4, [self.batch_size, s_h, s_w, self.c_dim],
                                    k_h=self.K_G_H3, 
                                    k_w=self.K_G_W3, 
                                    d_h=self.S_G_H, 
                                    d_w=self.S_G_W, 
                                    stddev = self.learn_rate_G,
                                    name='g_h5_con'))

      return h5

  def load_Raw(self):
    freqs_list_train, x_train, y_train, x_test, y_test = RFDataSetRawActive.load(self.freq_num, self.raw_sample_num, self.figure_dir)
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0).astype(np.int)
    seed = int((time.time()- int(time.time()))*10000)
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    return freqs_list_train, x/255., y_vec

  def load_sine_wave(self):
    sample_num = self.freq_num

    sine_num = 2000
    mu, sigma = 0, 0.1 
    Fs = 800
    f = 5
    samples = np.arange(sample_num)
    x_0 = np.zeros((sine_num, 1, sample_num, 1))
    for i in range(sine_num):
       sine_noise_0 = ((np.sin(0.5 * np.pi * f * np.arange(sample_num) / Fs) + np.random.normal(mu, sigma, sample_num)/10) )/2 + 0.52
       sine_noise_0 = np.reshape(sine_noise_0, (len(sine_noise_0),1))
       x_0[i] = sine_noise_0
    y_0 = np.zeros((sine_num, 2))
    y_0[:,0] = 1

    x_1 = np.zeros((sine_num, 1, sample_num, 1))
    for i in range(sine_num):
        sine_noise_1 = (np.sin(1 * np.pi * f * np.arange(sample_num) / Fs) + np.random.normal(mu, sigma, sample_num)/10 )/2 + 0.53
        sine_noise_1 = np.reshape(sine_noise_1, (len(sine_noise_1),1))
        x_1[i] = sine_noise_1
    y_1 = np.zeros((sine_num, 2))
    y_1[:,1] = 1

    x = np.concatenate((x_0,x_1))
    y_vec = np.concatenate((y_0,y_1))

    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s]
    y_vec = y_vec[s]
    return samples, x, y_vec

  def load_Power_Avg(self, location):
    
    #return self.load_sine_wave()

    freqs_list_train, x_train, y_train, x_test, y_test = RFDataSetPowerAvg.load(self.freq_num, self.figure_dir, location)

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0).astype(np.int)
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)

    x = x[randomize]
    y = y[randomize]

    x_Yes = x[np.where(y==1)[0]]
    y_Yes = y[np.where(y==1)[0]]

    x_No = x[np.where(y==0)[0]]
    y_No = y[np.where(y==0)[0]]

    print(len(x_Yes))
    print(len(x_No))

    train_data_file_name = os.path.join(self.train_data_dir, 'real_'+location+hd.Raw_Data_File_Name_Extension)
    with open(train_data_file_name, 'wb') as f:
        pickle.dump((freqs_list_train,x,y), f, pickle.HIGHEST_PROTOCOL)
    
    #x= x.reshape((len(y), 1, len(freqs_list_train),1))/abs(np.min(x))
    x_Yes= x_Yes.reshape((len(y_Yes), 1, len(freqs_list_train),1))
    x_No= x_No.reshape((len(y_No), 1, len(freqs_list_train),1))
    
    y_Yes_vec = np.zeros((len(y_Yes), self.y_dim), dtype=np.float)
    y_No_vec = np.zeros((len(y_No), self.y_dim), dtype=np.float)

    y_Yes_vec[:,1] = 1
    y_No_vec[:,0] = 1

    return freqs_list_train, x_Yes, y_Yes_vec, x_No, y_No_vec

  def load_Power_Avg_0917(self):
    
    #return self.load_sine_wave()

    freqs_list_train, x_train, y_train, x_test, y_test = RFDataSetPowerAvg.load(self.freq_num, self.figure_dir)
    #x = x_test[0:20]
    #y = y_test[0:20]

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0).astype(np.int)
    
    train_data_file_name = os.path.join(self.train_data_dir, 'real'+hd.Raw_Data_File_Name_Extension)
    with open(train_data_file_name, 'wb') as f:
        pickle.dump((freqs_list_train,x,y), f, pickle.HIGHEST_PROTOCOL)
    
    #x= x.reshape((len(y), 1, len(freqs_list_train),1))/abs(np.min(x))
    x= x.reshape((len(y), 1, len(freqs_list_train),1))
    
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]

    print(x.shape)
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0

    return freqs_list_train, x, y_vec
    
  def load_mnist(self):
    #data_dir = os.path.join(self.data_dir, self.dataset_name)
    data_dir = '/media/win/Code/RFResearch/GANIn/mnist'

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    print(self.dataset)
    train_num = 60000
    test_num = 10000
    #train_num = 6000
    #test_num = 1000
    if self.dataset == Dataset.MNIST:
      trX = (loaded[16:].reshape((60000,28,28,1)).astype(np.float))[:train_num]
    else:
      trX = (loaded[16:].reshape((60000,1,784,1)).astype(np.float))[:train_num]

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = (loaded[8:].reshape((60000)).astype(np.float))[:train_num]

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    if self.dataset == Dataset.MNIST:
      teX = (loaded[16:].reshape((10000,28,28,1)).astype(np.float))[:test_num]
    else:
      teX = (loaded[16:].reshape((10000,1,784,1)).astype(np.float))[:test_num]

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = (loaded[8:].reshape((10000)).astype(np.float))[:test_num]

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = int((time.time()- int(time.time()))*10000)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    return X/255.,y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(self.dataset, self.batch_size, self.output_height, self.output_width)

  def save(self, checkpoint_dir, step, filename='model', ckpt=True, frozen=False):
    # model_name = "DCGAN.model"
    # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    filename += '.b' + str(self.batch_size)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    if ckpt:
      self.saver.save(self.sess,
              os.path.join(checkpoint_dir, filename),
              global_step=step)

    if frozen:
      tf.train.write_graph(
              tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["generator_1/Tanh"]),
              checkpoint_dir,
              '{}-{:06d}_frz.pb'.format(filename, step),
              as_text=False)

  def load_model(self, checkpoint_dir):
    print(" [*] Reading checkpoints...", checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(ckpt_name.split('-')[-1])
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0