"""
Created by Bing Liu
Common functions used to build GAN
"""
from __future__ import division
import math
import pickle
import json
import random
import pprint
import scipy.misc
import cv2
import numpy as np
import os
import time
import datetime
from time import gmtime, strftime
from six.moves import xrange
from PIL import Image
from enum import Enum
import tensorflow as tf
import tensorflow.contrib.slim as slim
import RFDataSetRawActive
import Com_Power
import matplotlib.pyplot as plt
import Header as hd
import GAN_Model

class Dataset(Enum):
  MNIST = 1
  MNIST_1D = 2
  Raw = 3
  Power_Avg = 4

Lable_fontsize=10
pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def draw_loss(Figure_File_Name, loss_D, loss_G):
  x = range(len(loss_D))
  fig = plt.figure()
  plt.plot( x, loss_D, linewidth=0.5, label='Discriminator')
  plt.plot( x, loss_G, linewidth=0.5, label='Generator')
  fig.legend(loc = 'upper center', bbox_to_anchor=(0.5, 0.88))  
  plt.xlabel('Number of Training',fontsize=Lable_fontsize)
  plt.ylabel('Loss',fontsize=Lable_fontsize)
  plt.savefig(Figure_File_Name, dpi =400)
  plt.close()

def get_loss(csv_reader):
  loss_D = []
  loss_G = []
  for row in csv_reader:
    loss_D.append(float(row.split(',')[2]))
    loss_G.append(float(row.split(',')[4]))
  return loss_D, loss_G

def expand_path(path):
  return os.path.expanduser(os.path.expandvars(path))

def timestamp(s='%Y%m%d.%H%M%S', ts=None):
  if not ts: ts = time.time()
  st = datetime.datetime.fromtimestamp(ts).strftime(s)
  return st
  
def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    img_bgr = cv2.imread(path)
    img_rgb = img_bgr[..., ::-1]
    return img_rgb.astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def Visualize_Img(sess, dcgan, config, option, sample_dir='samples'):
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(sample_dir, 'test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime() )))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      y = np.random.choice(10, config.batch_size)
      y_one_hot = np.zeros((config.batch_size, 10))
      y_one_hot[np.arange(config.batch_size), y] = 1

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})

      if config.dataset == 'mnist_1d':
        samples = samples.reshape((config.batch_size, 28, 28, 1))

      save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(sample_dir, 'test_arange_%s.png' % (idx)))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, dcgan.z_dim - 1) for _ in xrange(dcgan.z_dim)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      y = np.random.choice(10, config.batch_size)
      y_one_hot = np.zeros((config.batch_size, 10))
      y_one_hot[np.arange(config.batch_size), y] = 1

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})

      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(sample_dir, 'test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime() )))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, os.path.join(sample_dir, 'test_gif_%s.gif' % (idx)))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], os.path.join(sample_dir, 'test_gif_%s.gif' % (idx)))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)

def Visualize_Raw(sess, dcgan, config, sample_dir='samples'):
  values = np.arange(0, 1, 1./config.batch_size)
  for idx in xrange(dcgan.z_dim):
    print(" [*] %d" % idx)
    z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
    for kdx, z in enumerate(z_sample):
      z[idx] = values[kdx]
    
    random.seed( int((time.time()- int(time.time()))*10000))    
    y = np.random.randint(0,2, config.batch_size)
    y_one_hot = np.zeros((config.batch_size, 2))
    y_one_hot[np.arange(config.batch_size), y] = 1
    Com_Power.Draw_Freqs_Powers_Labels_By_Raw(figure_file_name, np.array(dcgan.freqs_list_train), samples, y)

def Visualize_Power_Avg(sess, dcgan, config, sample_dir, generated_data_dir, train_location, test_location):
  freqs_list_train, data_X_Yes, data_y_Yes, data_X_No, data_y_No = dcgan.load_Power_Avg(test_location)

    
  values = np.arange(0, 1, 1./config.batch_size)
  
  gene_sample_num = len(data_X_No)
  gene_x = np.zeros((gene_sample_num,dcgan.z_dim), dtype=np.float)
  gene_y = np.ones(gene_sample_num, dtype=np.int)

  generated_data_file_name = os.path.join(generated_data_dir, ('gene_power_'+ test_location+ hd.Raw_Data_File_Name_Extension))
  figure_dir = os.path.join(generated_data_dir, 'gene_data_figures_train_{}_test_{}'.format(train_location, test_location))
  if not os.path.exists(figure_dir): os.makedirs(figure_dir)
  for idx in xrange(int(gene_sample_num/config.batch_size)):
    print(" [*] %d" % idx)
    z_sample = np.reshape( data_X_No[idx*config.batch_size:(idx+1)*config.batch_size ], (config.batch_size , dcgan.z_dim))
    y = np.ones(config.batch_size, dtype=np.int)
    y_one_hot = np.zeros((config.batch_size, 2))
    y_one_hot[np.arange(config.batch_size), y] = 1

    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
    gene_x[idx*config.batch_size:(idx+1)*config.batch_size] = np.reshape(samples, (config.batch_size,dcgan.z_dim))*-40
    figure_file_name = '{}/test_{:08d}.png'.format(figure_dir, idx)
    Com_Power.Draw_Freqs_Figure_Single_Sub_Plot_Label(figure_file_name, dcgan.freqs_list_train, -40*samples.reshape((dcgan.batch_size,-1)), y, dcgan.batch_size, 1)

    for i in range(dcgan.batch_size):
      figure_file_name = '{}/test_{:08d}_{}.png'.format(figure_dir, idx, i)
      Com_Power.Draw_Freqs_Figure_One(figure_file_name, dcgan.freqs_list_train, -40*samples[i].reshape(len(dcgan.freqs_list_train)), line_color='gray',my_figsize=(8, 4))


  with open(generated_data_file_name, 'wb') as f:
      pickle.dump((dcgan.freqs_list_train, gene_x, gene_y), f, pickle.HIGHEST_PROTOCOL)

def Visualize_Power_Avg_0917(sess, dcgan, config, sample_dir, generated_data_dir, train_data_dir):
  values = np.arange(0, 1, 1./config.batch_size)
  gene_x = np.zeros((dcgan.gene_sample_num*config.batch_size,dcgan.z_dim), dtype=np.float)
  gene_y = np.zeros(dcgan.gene_sample_num*config.batch_size, dtype=np.int)
  generated_data_file_name = os.path.join(generated_data_dir, ('gene_power'+ hd.Raw_Data_File_Name_Extension))
  figure_dir = os.path.join(generated_data_dir, 'gene_data_figures')
  if not os.path.exists(figure_dir): os.makedirs(figure_dir)
  for idx in xrange(dcgan.gene_sample_num):
    print(" [*] %d" % idx)
    z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
    
    random.seed( int((time.time()- int(time.time()))*10000))    
    y = np.random.randint(0,2, config.batch_size)
    gene_y[idx*config.batch_size:(idx+1)*config.batch_size] = y
    y_one_hot = np.zeros((config.batch_size, 2))
    y_one_hot[np.arange(config.batch_size), y] = 1

    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
    gene_x[idx*config.batch_size:(idx+1)*config.batch_size] = np.reshape(samples, (config.batch_size,dcgan.z_dim))*-40

  with open(generated_data_file_name, 'wb') as f:
      pickle.dump((dcgan.freqs_list_train, gene_x, gene_y), f, pickle.HIGHEST_PROTOCOL)

def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w

def Create_Output_Folder_Name_Power(Freq_Num,
                                    kernel_D_H0,
                                    kernel_D_W0,
                                    kernel_D_H1,
                                    kernel_D_W1,
                                    kernel_D_H2,
                                    kernel_D_W2,
                                    kernel_G_H0,
                                    kernel_G_W0,
                                    kernel_G_H1,
                                    kernel_G_W1,
                                    kernel_G_H2,
                                    kernel_G_W2,
                                    kernel_G_H3,
                                    kernel_G_W3,
                                    Stride_D_H,
                                    Stride_D_W,
                                    Stride_G_H,
                                    Stride_G_W,
                                    train_location,
                                    test_location):
  return "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_Model_Loc_{17}".format(
                                                      Freq_Num,
                                                      kernel_D_H0,
                                                      kernel_D_W0,
                                                      kernel_D_H1,
                                                      kernel_D_W1,
                                                      kernel_D_H2,
                                                      kernel_D_W2,
                                                      kernel_G_H0,
                                                      kernel_G_W0,
                                                      kernel_G_H1,
                                                      kernel_G_W1,
                                                      kernel_G_H2,
                                                      kernel_G_W2,
                                                      kernel_G_H3,
                                                      kernel_G_W3,
                                                      Stride_D_H,
                                                      Stride_D_W,
                                                      Stride_G_H,
                                                      Stride_G_W,
                                                      train_location)

def Create_Output_Folder_Name_Rawold( Freq_Num,
                                    Sample_Num,
                                    kernel_D_H,
                                    kernel_D_W,
                                    kernel_G_H,
                                    kernel_G_W,
                                    Stride_D_H,
                                    Stride_D_W,
                                    Stride_G_H,
                                    Stride_G_W):
  return "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}".format(Freq_Num,
                                                      kernel_D_H,
                                                      kernel_D_W,
                                                      kernel_G_H,
                                                      kernel_G_W,
                                                      Stride_D_H,
                                                      Stride_D_W,
                                                      Stride_G_H,
                                                      Stride_G_W)
def Create_Output_Folder_Name_Raw(Freq_Num,
                                    Sample_Num,
                                    kernel_D_H0,
                                    kernel_D_W0,
                                    kernel_D_H1,
                                    kernel_D_W1,
                                    kernel_D_H2,
                                    kernel_D_W2,
                                    kernel_G_H0,
                                    kernel_G_W0,
                                    kernel_G_H1,
                                    kernel_G_W1,
                                    kernel_G_H2,
                                    kernel_G_W2,
                                    kernel_G_H3,
                                    kernel_G_W3,
                                    Stride_D_H,
                                    Stride_D_W,
                                    Stride_G_H,
                                    Stride_G_W,
                                    train_location,
                                    test_location):
  return "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_Model_Loc_{17}".format(
                                                      Freq_Num,
                                                      kernel_D_H0,
                                                      kernel_D_W0,
                                                      kernel_D_H1,
                                                      kernel_D_W1,
                                                      kernel_D_H2,
                                                      kernel_D_W2,
                                                      kernel_G_H0,
                                                      kernel_G_W0,
                                                      kernel_G_H1,
                                                      kernel_G_W1,
                                                      kernel_G_H2,
                                                      kernel_G_W2,
                                                      kernel_G_H3,
                                                      kernel_G_W3,
                                                      Stride_D_H,
                                                      Stride_D_W,
                                                      Stride_G_H,
                                                      Stride_G_W,
                                                      train_location)



def Create_Output_Folder_Name_Img(  
                                    kernel_D_H0,
                                    kernel_D_W0,
                                    kernel_D_H1,
                                    kernel_D_W1,
                                    kernel_D_H2,
                                    kernel_D_W2,
                                    kernel_G_H0,
                                    kernel_G_W0,
                                    kernel_G_H1,
                                    kernel_G_W1,
                                    kernel_G_H2,
                                    kernel_G_W2,
                                    Stride_D_H,
                                    Stride_D_W,
                                    Stride_G_H,
                                    Stride_G_W):
  return "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}".format(
                                                      kernel_D_H0,
                                                      kernel_D_W0,
                                                      kernel_D_H1,
                                                      kernel_D_W1,
                                                      kernel_D_H2,
                                                      kernel_D_W2,
                                                      kernel_G_H0,
                                                      kernel_G_W0,
                                                      kernel_G_H1,
                                                      kernel_G_W1,
                                                      kernel_G_H2,
                                                      kernel_G_W2,
                                                      Stride_D_H,
                                                      Stride_D_W,
                                                      Stride_G_H,
                                                      Stride_G_W)
