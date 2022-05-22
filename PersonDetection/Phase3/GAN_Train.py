"""
Created by Bing Liu
Train GAN model
"""
import os
import scipy.misc
import numpy as np
import tensorflow as tf
import json
from GAN_Model import DCGAN
from Com_GAN import *

dataset = Dataset.Raw
Root = 'e:/Code/RFResearch/'
ISTrain = True

Freq_num = 784
Raw_Sample_Num = 4800

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "")
flags.DEFINE_boolean("train", ISTrain, "")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")

if dataset == Dataset.MNIST:
  y_dim = 10
  flags.DEFINE_integer("batch_size", 64, "")
  flags.DEFINE_float("learn_rate_D", 0.002, "")
  flags.DEFINE_float("learn_rate_G", 0.002, "")
  flags.DEFINE_integer("input_height", 28, "")
  flags.DEFINE_integer("input_width", None, "")
  flags.DEFINE_integer("output_height", 28, "")
  flags.DEFINE_integer("output_width", None, "")
  flags.DEFINE_string("dataset", "mnist", "")
  flags.DEFINE_integer("K_D_H0", 3, "")
  flags.DEFINE_integer("K_D_W0", 3, "")
  flags.DEFINE_integer("K_D_H1", 3, "")
  flags.DEFINE_integer("K_D_W1", 3, "")
  flags.DEFINE_integer("K_D_H2", 5, "")
  flags.DEFINE_integer("K_D_W2", 5, "")
  flags.DEFINE_integer("K_D_H3", 5, "")
  flags.DEFINE_integer("K_D_W3", 5, "")
  flags.DEFINE_integer("S_D_H", 2, "")
  flags.DEFINE_integer("S_D_W", 2, "")
  flags.DEFINE_integer("K_G_H0", 3, "")
  flags.DEFINE_integer("K_G_W0", 3, "")
  flags.DEFINE_integer("K_G_H1", 3, "")
  flags.DEFINE_integer("K_G_W1", 3, "")
  flags.DEFINE_integer("K_G_H2", 5, "")
  flags.DEFINE_integer("K_G_W2", 5, "")
  flags.DEFINE_integer("K_G_H3", 5, "")
  flags.DEFINE_integer("K_G_W3", 5, "")
  flags.DEFINE_integer("S_G_H", 2, "")
  flags.DEFINE_integer("S_G_W", 2, "")
  #flags.DEFINE_integer("gf_dim", 64, "")
  #flags.DEFINE_integer("df_dim", 64, "")
  flags.DEFINE_integer("gf_dim", 32, "")
  flags.DEFINE_integer("df_dim", 32, "")
  flags.DEFINE_integer("gfc_dim", 1024, "")
  flags.DEFINE_integer("dfc_dim", 1024, "")
  #flags.DEFINE_integer("gfc_dim", 512, "")
  #flags.DEFINE_integer("dfc_dim", 512, "")

elif dataset == Dataset.MNIST_1D:
  y_dim = 10
  flags.DEFINE_integer("batch_size", 64, "")
  flags.DEFINE_float("learn_rate_D", 0.0002, "")
  flags.DEFINE_float("learn_rate_G", 0.0002, "")
  flags.DEFINE_integer("input_height", 1, "")
  flags.DEFINE_integer("input_width", 784, "")
  flags.DEFINE_integer("output_height", 1, "")
  flags.DEFINE_integer("output_width", 784, "")
  flags.DEFINE_string("dataset", "mnist_1d", "")
  flags.DEFINE_integer("K_D_H0", 1, "")
  flags.DEFINE_integer("K_D_W0", 3, "")
  flags.DEFINE_integer("K_D_H1", 1, "")
  flags.DEFINE_integer("K_D_W1", 5, "")
  flags.DEFINE_integer("K_D_H2", 1, "")
  flags.DEFINE_integer("K_D_W2", 5, "")
  flags.DEFINE_integer("K_D_H3", 1, "")
  flags.DEFINE_integer("K_D_W3", 5, "")
  flags.DEFINE_integer("S_D_H", 1, "")
  flags.DEFINE_integer("S_D_W", 2, "")
  flags.DEFINE_integer("K_G_H0", 1, "")
  flags.DEFINE_integer("K_G_W0", 3, "")
  flags.DEFINE_integer("K_G_H1", 1, "")
  flags.DEFINE_integer("K_G_W1", 5, "")
  flags.DEFINE_integer("K_G_H2", 1, "")
  flags.DEFINE_integer("K_G_W2", 5, "")
  flags.DEFINE_integer("K_G_H3", 1, "")
  flags.DEFINE_integer("K_G_W3", 5, "")
  flags.DEFINE_integer("S_G_H", 1, "")
  flags.DEFINE_integer("S_G_W", 2, "")
  #flags.DEFINE_integer("gf_dim", 64, "")
  #flags.DEFINE_integer("df_dim", 64, "")
  #flags.DEFINE_integer("gfc_dim", 1024, "")
  #flags.DEFINE_integer("dfc_dim", 1024, "")
  flags.DEFINE_integer("gf_dim", 32, "")
  flags.DEFINE_integer("df_dim", 32, "")
  flags.DEFINE_integer("gfc_dim", 512, "")
  flags.DEFINE_integer("dfc_dim", 512, "")
elif dataset == Dataset.Raw:
  y_dim = 2
  flags.DEFINE_integer("gene_sample_num", 800, "")
  flags.DEFINE_integer("batch_size", 2, "")
  flags.DEFINE_float("learn_rate_D", 0.0002, "")
  flags.DEFINE_float("learn_rate_G", 0.0002, "")
  flags.DEFINE_integer("input_height", Freq_num, "")
  flags.DEFINE_integer("input_width", Raw_Sample_Num, "")
  flags.DEFINE_integer("output_height", Freq_num, "")
  flags.DEFINE_integer("output_width", Raw_Sample_Num, "")
  flags.DEFINE_string("dataset", "raw", "")
  flags.DEFINE_string("train_location", "01", "")
  flags.DEFINE_string("test_location", "01", "")
  flags.DEFINE_integer("K_D_H0", 1, "")
  flags.DEFINE_integer("K_D_W0", 3, "")
  flags.DEFINE_integer("K_D_H1", 1, "")
  flags.DEFINE_integer("K_D_W1", 3, "")
  flags.DEFINE_integer("K_D_H2", 1, "")
  flags.DEFINE_integer("K_D_W2", 5, "")
  flags.DEFINE_integer("K_D_H3", 0, "")
  flags.DEFINE_integer("K_D_W3", 0, "")
  flags.DEFINE_integer("S_D_H", 1, "")
  flags.DEFINE_integer("S_D_W", 2, "")
  flags.DEFINE_integer("K_G_H0", 1, "")
  flags.DEFINE_integer("K_G_W0", 3, "")
  flags.DEFINE_integer("K_G_H1", 1, "")
  flags.DEFINE_integer("K_G_W1", 3, "")
  flags.DEFINE_integer("K_G_H2", 1, "")
  flags.DEFINE_integer("K_G_W2", 5, "")
  flags.DEFINE_integer("K_G_H3", 0, "")
  flags.DEFINE_integer("K_G_W3", 0, "")
  flags.DEFINE_integer("S_G_H", 1, "")
  flags.DEFINE_integer("S_G_W", 2, "")
  flags.DEFINE_integer("gf_dim", 100, "")
  flags.DEFINE_integer("df_dim", 100, "")
  flags.DEFINE_integer("gfc_dim", 400, "")
  flags.DEFINE_integer("dfc_dim", 400, "")

else:
  y_dim = 2
  flags.DEFINE_integer("gene_sample_num", 800, "")
  flags.DEFINE_integer("batch_size", 4, "")
  #flags.DEFINE_float("learn_rate_D", 0.0005, "") #power
  #flags.DEFINE_float("learn_rate_G", 0.0005, "") #power
  #flags.DEFINE_float("learn_rate_D", 0.005, "")  #sine1
  #flags.DEFINE_float("learn_rate_G", 0.005, "")  #sine1
  flags.DEFINE_float("learn_rate_D", 0.0005, "") 
  flags.DEFINE_float("learn_rate_G", 0.0005, "") 
  flags.DEFINE_integer("input_height", 1, "")
  flags.DEFINE_integer("input_width", Freq_num, "")
  flags.DEFINE_integer("output_height", 1, "")
  flags.DEFINE_integer("output_width", Freq_num, "")
  flags.DEFINE_string("dataset", "power_avg", "")
  flags.DEFINE_string("train_location", "01", "")
  #flags.DEFINE_string("test_location", "01", "")
  #flags.DEFINE_string("test_location", "05", "")
  #flags.DEFINE_string("test_location", "07", "")
  #flags.DEFINE_string("test_location", "06", "")
  flags.DEFINE_string("test_location", "09", "")
  flags.DEFINE_integer("K_D_H0", 1, "")
  flags.DEFINE_integer("K_D_W0", 3, "")
  flags.DEFINE_integer("K_D_H1", 1, "")
  flags.DEFINE_integer("K_D_W1", 3, "")
  flags.DEFINE_integer("K_D_H2", 1, "")
  flags.DEFINE_integer("K_D_W2", 5, "")
  flags.DEFINE_integer("K_D_H3", 0, "")
  flags.DEFINE_integer("K_D_W3", 0, "")
  flags.DEFINE_integer("S_D_H", 1, "")
  flags.DEFINE_integer("S_D_W", 2, "")
  flags.DEFINE_integer("K_G_H0", 1, "")
  flags.DEFINE_integer("K_G_W0", 3, "")
  flags.DEFINE_integer("K_G_H1", 1, "")
  flags.DEFINE_integer("K_G_W1", 3, "")
  flags.DEFINE_integer("K_G_H2", 1, "")
  flags.DEFINE_integer("K_G_W2", 5, "")
  flags.DEFINE_integer("K_G_H3", 0, "")
  flags.DEFINE_integer("K_G_W3", 0, "")
  flags.DEFINE_integer("S_G_H", 1, "")
  flags.DEFINE_integer("S_G_W", 2, "")
  flags.DEFINE_integer("gf_dim", 100, "")
  flags.DEFINE_integer("df_dim", 100, "")
  flags.DEFINE_integer("gfc_dim", 400, "")
  flags.DEFINE_integer("dfc_dim", 400, "")
  #flags.DEFINE_integer("gf_dim", 128, "")
  #flags.DEFINE_integer("df_dim", 128, "")
  #flags.DEFINE_integer("gfc_dim", 1024, "")
  #flags.DEFINE_integer("dfc_dim", 1024, "")

flags.DEFINE_string("input_fname_pattern", "*.jpg", "")
flags.DEFINE_string("data_dir", "./data", "")
flags.DEFINE_string("out_dir", Root + "GANout", "")
#flags.DEFINE_string("data_dir", "E:\Code\RFResearch\GAN\data", "path to datasets [e.g. $HOME/data]")
#flags.DEFINE_string("out_dir", "E:\Code\RFResearch\GAN\out", "Root directory for outputs [e.g. $HOME/out]")

flags.DEFINE_string("out_name", "", "")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "")
flags.DEFINE_string("loss_dir", "loss", "")
flags.DEFINE_string("generated_data_dir", "generated_data", "")
flags.DEFINE_string("train_data_dir", "real_data", "")
flags.DEFINE_string("sample_dir", "samples", "")
flags.DEFINE_string("figure_dir", "figures", "")
flags.DEFINE_boolean("visualize", True, "")
flags.DEFINE_boolean("export", False, "True for exporting with new batch size")
flags.DEFINE_boolean("freeze", False, "True for exporting with new batch size")
flags.DEFINE_integer("max_to_keep", 1, "maximum number of checkpoints to keep")
if dataset == Dataset.MNIST or  dataset == Dataset.MNIST_1D:
  flags.DEFINE_integer("sample_freq", 200, "")
  flags.DEFINE_integer("ckpt_freq", 200, "")
  flags.DEFINE_integer("z_dim", 784, "")
elif dataset == Dataset.Raw:
  flags.DEFINE_integer("sample_freq", 40, "")
  flags.DEFINE_integer("ckpt_freq", 40, "")
  flags.DEFINE_integer("z_dim", 9600, "")
else:
  flags.DEFINE_integer("sample_freq", 10, "")
  flags.DEFINE_integer("ckpt_freq", 40, "")
  flags.DEFINE_integer("z_dim", 784, "")

flags.DEFINE_string("z_dist", "uniform_signed", "'normal01' or 'uniform_unsigned' or uniform_signed")
flags.DEFINE_boolean("G_img_sum", False, "Save generator image summaries in log")
#flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  
  # expand user name and environment variables
  FLAGS.data_dir = expand_path(FLAGS.data_dir)
  FLAGS.out_dir = expand_path(FLAGS.out_dir)
  FLAGS.out_name = expand_path(FLAGS.out_name)
  FLAGS.checkpoint_dir = expand_path(FLAGS.checkpoint_dir)
  FLAGS.loss_dir = expand_path(FLAGS.loss_dir)
  FLAGS.sample_dir = expand_path(FLAGS.sample_dir)
  FLAGS.generated_data_dir = expand_path(FLAGS.generated_data_dir)
  FLAGS.train_data_dir = expand_path(FLAGS.train_data_dir)
  FLAGS.figure_dir = expand_path(FLAGS.figure_dir)

  if FLAGS.output_height is None: FLAGS.output_height = FLAGS.input_height
  if FLAGS.input_width is None: FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None: FLAGS.output_width = FLAGS.output_height

  if dataset == Dataset.MNIST:
    FLAGS.out_name = 'MNIST_2D_{}'.format(Create_Output_Folder_Name_Img(
                                                                        FLAGS.K_D_H0,
                                                                        FLAGS.K_D_W0,
                                                                        FLAGS.K_D_H1,
                                                                        FLAGS.K_D_W1,
                                                                        FLAGS.K_D_H2,
                                                                        FLAGS.K_D_W2,
                                                                        FLAGS.S_D_H,
                                                                        FLAGS.S_D_W,
                                                                        FLAGS.K_G_H0,
                                                                        FLAGS.K_G_W0,
                                                                        FLAGS.K_G_H1,
                                                                        FLAGS.K_G_W1,
                                                                        FLAGS.K_G_H2,
                                                                        FLAGS.K_G_W2,
                                                                        FLAGS.S_G_H,
                                                                        FLAGS.S_G_W))
  elif dataset == Dataset.MNIST_1D:
    FLAGS.out_name = 'MNIST_1D_{}'.format(Create_Output_Folder_Name_Img(
                                                                        FLAGS.K_D_H0,
                                                                        FLAGS.K_D_W0,
                                                                        FLAGS.K_D_H1,
                                                                        FLAGS.K_D_W1,
                                                                        FLAGS.K_D_H2,
                                                                        FLAGS.K_D_W2,
                                                                        FLAGS.S_D_H,
                                                                        FLAGS.S_D_W,
                                                                        FLAGS.K_G_H0,
                                                                        FLAGS.K_G_W0,
                                                                        FLAGS.K_G_H1,
                                                                        FLAGS.K_G_W1,
                                                                        FLAGS.K_G_H2,
                                                                        FLAGS.K_G_W2,
                                                                        FLAGS.S_G_H,
                                                                        FLAGS.S_G_W))
    #FLAGS.out_name = 'MNIST_1D'

  elif dataset == Dataset.Raw:
    FLAGS.out_name = 'Raw_{}'.format(Create_Output_Folder_Name_Raw( Freq_num,
                                                                      Raw_Sample_Num,
                                                                            FLAGS.K_D_H0,
                                                                            FLAGS.K_D_W0,
                                                                            FLAGS.K_D_H1,
                                                                            FLAGS.K_D_W1,
                                                                            FLAGS.K_D_H2,
                                                                            FLAGS.K_D_W2,
                                                                            FLAGS.S_D_H,
                                                                            FLAGS.S_D_W,
                                                                            FLAGS.K_G_H0,
                                                                            FLAGS.K_G_W0,
                                                                            FLAGS.K_G_H1,
                                                                            FLAGS.K_G_W1,
                                                                            FLAGS.K_G_H2,
                                                                            FLAGS.K_G_W2,
                                                                            FLAGS.K_G_H3,
                                                                            FLAGS.K_G_W3,
                                                                            FLAGS.S_G_H,
                                                                            FLAGS.S_G_W,
                                                                            FLAGS.train_location,
                                                                            FLAGS.test_location))
  else:
    FLAGS.out_name = 'Power_Avg_{}'.format(Create_Output_Folder_Name_Power( Freq_num,
                                                                            FLAGS.K_D_H0,
                                                                            FLAGS.K_D_W0,
                                                                            FLAGS.K_D_H1,
                                                                            FLAGS.K_D_W1,
                                                                            FLAGS.K_D_H2,
                                                                            FLAGS.K_D_W2,
                                                                            FLAGS.S_D_H,
                                                                            FLAGS.S_D_W,
                                                                            FLAGS.K_G_H0,
                                                                            FLAGS.K_G_W0,
                                                                            FLAGS.K_G_H1,
                                                                            FLAGS.K_G_W1,
                                                                            FLAGS.K_G_H2,
                                                                            FLAGS.K_G_W2,
                                                                            FLAGS.K_G_H3,
                                                                            FLAGS.K_G_W3,
                                                                            FLAGS.S_G_H,
                                                                            FLAGS.S_G_W,
                                                                            FLAGS.train_location,
                                                                            FLAGS.test_location))
  FLAGS.out_dir = os.path.join(FLAGS.out_dir, FLAGS.out_name)
  FLAGS.checkpoint_dir = os.path.join(FLAGS.out_dir, FLAGS.checkpoint_dir)
  FLAGS.loss_dir = os.path.join(FLAGS.out_dir, FLAGS.loss_dir)
  FLAGS.sample_dir = os.path.join(FLAGS.out_dir, FLAGS.sample_dir)
  FLAGS.generated_data_dir = os.path.join(FLAGS.out_dir, FLAGS.generated_data_dir)
  FLAGS.train_data_dir = os.path.join(FLAGS.out_dir, FLAGS.train_data_dir)
  FLAGS.figure_dir = os.path.join(FLAGS.out_dir, FLAGS.figure_dir)

  if not os.path.exists(FLAGS.checkpoint_dir): os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.loss_dir): os.makedirs(FLAGS.loss_dir)
  if not os.path.exists(FLAGS.sample_dir): os.makedirs(FLAGS.sample_dir)
  if not os.path.exists(FLAGS.train_data_dir): os.makedirs(FLAGS.train_data_dir)
  if not os.path.exists(FLAGS.generated_data_dir): os.makedirs(FLAGS.generated_data_dir)
  if not os.path.exists(FLAGS.figure_dir): os.makedirs(FLAGS.figure_dir)

  with open(os.path.join(FLAGS.out_dir, 'FLAGS.json'), 'w') as f:
    flags_dict = {k:FLAGS[k].value for k in FLAGS}
    json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
  

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  
  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(sess,
                  gene_sample_num=FLAGS.gene_sample_num,
                  input_width=FLAGS.input_width,
                  input_height=FLAGS.input_height,
                  output_width=FLAGS.output_width,
                  output_height=FLAGS.output_height,
                  batch_size=FLAGS.batch_size,
                  learn_rate_D=FLAGS.learn_rate_D,
                  learn_rate_G=FLAGS.learn_rate_G,
                  y_dim=y_dim,
                  z_dim=FLAGS.z_dim,
                  K_D_H0=FLAGS.K_D_H0,
                  K_D_W0=FLAGS.K_D_W0,
                  K_D_H1=FLAGS.K_D_H1,
                  K_D_W1=FLAGS.K_D_W1,
                  K_D_H2=FLAGS.K_D_H2,
                  K_D_W2=FLAGS.K_D_W2,
                  K_D_H3=FLAGS.K_D_H3,
                  K_D_W3=FLAGS.K_D_W3,
                  S_D_H=FLAGS.S_D_H,
                  S_D_W=FLAGS.S_D_W,
                  K_G_H0=FLAGS.K_G_H0,
                  K_G_W0=FLAGS.K_G_W0,
                  K_G_H1=FLAGS.K_G_H1,
                  K_G_W1=FLAGS.K_G_W1,
                  K_G_H2=FLAGS.K_G_H2,
                  K_G_W2=FLAGS.K_G_W2,
                  S_G_H=FLAGS.S_G_H,
                  S_G_W=FLAGS.S_G_W,
                  gf_dim=FLAGS.gf_dim,
                  df_dim=FLAGS.df_dim,
                  gfc_dim=FLAGS.gfc_dim,
                  dfc_dim=FLAGS.dfc_dim,
                  dataset = dataset,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  loss_dir=FLAGS.loss_dir,
                  sample_dir=FLAGS.sample_dir,
                  generated_data_dir=FLAGS.generated_data_dir,
                  train_data_dir=FLAGS.train_data_dir,
                  figure_dir=FLAGS.figure_dir,
                  data_dir=FLAGS.data_dir,
                  out_dir=FLAGS.out_dir,
                  max_to_keep=FLAGS.max_to_keep,
                  freq_num = Freq_num,
                  raw_sample_num = Raw_Sample_Num,
                  train_location = FLAGS.train_location,
                  test_location = FLAGS.test_location,)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      load_success, load_counter = dcgan.load_model(FLAGS.checkpoint_dir)
      if not load_success:
        raise Exception("Checkpoint not found in " + FLAGS.checkpoint_dir)

    # Below is codes for visualization
      if FLAGS.export:
        export_dir = os.path.join(FLAGS.checkpoint_dir, 'export_b'+str(FLAGS.batch_size))
        dcgan.save(export_dir, load_counter, ckpt=True, frozen=False)

      if FLAGS.freeze:
        export_dir = os.path.join(FLAGS.checkpoint_dir, 'frozen_b'+str(FLAGS.batch_size))
        dcgan.save(export_dir, load_counter, ckpt=False, frozen=True)

      if FLAGS.visualize:
        OPTION = 1
        if dataset == Dataset.MNIST or dataset == Dataset.MNIST_1D:
          Visualize_Img(sess, dcgan, FLAGS, OPTION, FLAGS.sample_dir)
        elif dataset == Dataset.Raw:
          Visualize_Raw(sess, dcgan, FLAGS, FLAGS.sample_dir)
        else:
          Visualize_Power_Avg(sess, dcgan, FLAGS, FLAGS.sample_dir, FLAGS.generated_data_dir, FLAGS.train_location,FLAGS.test_location)

if __name__ == '__main__':
  tf.app.run()