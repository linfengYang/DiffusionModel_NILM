import os
import random

# from S2S_Model import get_model,get_FPNmodel,get_FpnPanmodel
from EasyS2S_Model import get_model
from DataProvider import ChunkDoubleSourceSlider2, ChunkS2S_Slider
import NetFlowExt as nf
from Logger import log
from nilm_metric import *
# #import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

#############
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from Arguments import *

#REDD UK_DALE
originModel=False
datasetName='UK_DALE'
applianceName='kettle'
TrainNum=0
TrainPercent='20'
switch=1
def remove_space(string):
    return string.replace(" ", "")

print(TrainNum)
def set_seed(seed=2024):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

set_seed(2024)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network\
                                     for energy disaggregation - \
                                     network input = mains window; \
                                     network target nj= the states of \
                                     the target appliance.')
    parser.add_argument('--appliance_name',
                        type=remove_space,
                        default=f'{applianceName}', #-------------------kettle
                        help='the name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default=f'dataset_preprocess/created_data/{datasetName}/', #---------------
                        help='this is the directory of the training samples')
    parser.add_argument('--pretrainedmodel_dir',
                        type=str,
                        default='./pretrained_model',
                        help='this is the directory of the pre-trained models')
    parser.add_argument('--save_dir',
                        type=str,
                        default=f'./models/EasyS2S/{datasetName}',
                        help='this is the directory to save the trained models')
    parser.add_argument('--batchsize',
                        type=int,
                        default=1024,
                        help='The batch size of training examples')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=100,
                        help='The number of epochs.')
    parser.add_argument('--save_model',
                        type=int,
                        default=-1,
                        help='Save the learnt model:\
                        0 -- not to save the learnt model parameters;\
                        n (n>0) -- to save the model params every n steps;\
                        -1 -- only save the learnt model params\
                        at the end of training.')
    parser.add_argument('--dense_layers',
                        type=int,
                        default=1,
                        help=':\
                                1 -- One dense layers (default Seq2point);\
                                2 -- Two dense layers;\
                                3 -- Three dense layers.')
    parser.add_argument("--transfer_model", type=str2bool,
                        default=False,
                        help="True: using entire pre-trained model.\
                             False: retrain the entire pre-trained model;\
                             This will override the 'transfer_cnn' and 'cnn' parameters;\
                             The appliance_name parameter will use to retrieve \
                             the entire pre-trained model of that appliance.")
    parser.add_argument("--transfer_cnn", type=str2bool,
                        default=False,
                        help="True: using a pre-trained CNN\
                              False: not using a pre-trained CNN.")
    parser.add_argument('--cnn',
                        type=str,
                        default=f'{applianceName}', # ------------------
                        help='The CNN trained by which appliance to load (pretrained model).')
    parser.add_argument('--gpus',
                        type=int,
                        default=-1,
                        help='Number of GPUs to use:\
                            n -- number of GPUs the system should use;\
                            -1 -- do not use any GPU.')
    parser.add_argument('--crop_dataset',
                        type=int,
                        default=None,
                        help='for debugging porpose should be helpful to crop the training dataset size')
    parser.add_argument('--ram',
                        type=int,
                        default=5 * 10 ** 5,
                        help='Maximum number of rows of csv dataset can handle without loading in chunks')
    return parser.parse_args()


args = get_arguments()
log('Arguments: ')
log(args)

# some constant parameters
CHUNK_SIZE = 5 * 10 ** 6

# start the session for training a network
sess = tf.InteractiveSession()

# the appliance to train on
appliance_name = args.appliance_name

if originModel:
    trainfile = f'{applianceName}_{TrainPercent}training_'
else:
    #trainfile=f'{datasetName}Combined{applianceName}_fileEight'
    trainfile = f'{datasetName}Combined{applianceName}_file{TrainPercent}' #20
    # trainfile = 'LongCleanCombinedMicroWave_fileEight'

# path for training data
# training_path = args.datadir + appliance_name + '/' + appliance_name + '_training_' + '.csv'
training_path = args.datadir + appliance_name + '/' + trainfile + '.csv'
log('Training dataset: ' + training_path)

print(args.datadir)

# Looking for the validation set
for filename in os.listdir(args.datadir + appliance_name):
    if "validation" in filename:
        val_filename = filename
        log(val_filename)

# path for validation data
validation_path = args.datadir + appliance_name + '/' + val_filename
log('Validation dataset: ' + validation_path)

# offset parameter from window length
# offset = int(0.5*(params_appliance[args.appliance_name]['windowlength']-1.0))

windowlength = 600 # 599
# params_appliance[args.appliance_name]['windowlength']

# Defining object for training set loading and windowing provider (DataProvider.py)
train_provider = ChunkS2S_Slider(filename=training_path,
                                 batchsize=args.batchsize,  # default=1000
                                 chunksize=CHUNK_SIZE,  # 5*10**6
                                 crop=args.crop_dataset,
                                 shuffle=True,
                                 length=windowlength,  # 599
                                 header=0,
                                 ram_threshold=args.ram)  # ram  default=5*10**5

# Defining object for validation set loading and windowing provider (DataProvider.py)
val_provider = ChunkS2S_Slider(filename=validation_path,
                               batchsize=args.batchsize,
                               chunksize=CHUNK_SIZE,
                               crop=args.crop_dataset,
                               shuffle=False,
                               length=windowlength,
                               header=0,
                               ram_threshold=args.ram)

# TensorFlow placeholders

x = tf.placeholder(tf.float32,
                   shape=[None, windowlength],
                   name='x')

y_ = tf.placeholder(tf.float32,
                    shape=[None, windowlength],
                    name='y_')

# -------------------------------- Keras Network - from model.py -----------------------------------------

inp = Input(tensor=x)
model, cnn_check_weights = get_model(args.appliance_name,   #seq2seq model
                      inp,  # 预定义
                      windowlength,
                      transfer_dense=args.transfer_model,
                      transfer_cnn=args.transfer_cnn,
                      cnn=args.cnn,
                      pretrainedmodel_dir=args.pretrainedmodel_dir)
# cnn_check_weights
y = model.outputs


# #---FPN model----------------------
# inp = Input(tensor=x)
# model, cnn_check_weights = get_FPNmodel(args.appliance_name,  #FPN model
#                       inp,  # 预定义
#                       windowlength,
#                       transfer_dense=args.transfer_model,
#                       transfer_cnn=args.transfer_cnn,
#                       cnn=args.cnn,
#                       pretrainedmodel_dir=args.pretrainedmodel_dir)
# # cnn_check_weights
# #----------------------------------

#---------------------------#FPN-PAN model---------------------------
# inp = Input(tensor=x)
# model, cnn_check_weights = get_FpnPanmodel(args.appliance_name, 
#                       inp,  # 预定义
#                       windowlength,
#                       transfer_dense=args.transfer_model,
#                       transfer_cnn=args.transfer_cnn,
#                       cnn=args.cnn,
#                       pretrainedmodel_dir=args.pretrainedmodel_dir)
# # cnn_check_weights
# y = model.outputs  # 令模型输出为y  含有预定义的x
#-------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

# cost function
delta1=0.5
log(f'{delta1}---------------------',)
def huber_loss(y_true, y_pred, delta=delta1):
    residual = tf.abs(y_true - y_pred)
    condition = tf.less(residual, delta)
    small_res = tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)

    return tf.where(condition, small_res, large_res)

# 定义Log-Cosh损失函数
def log_cosh_loss(y_true, y_pred):
    def log_cosh(x):
        return x + tf.nn.softplus(-2.0 * x) - tf.math.log(2.0)
    return tf.reduce_mean(log_cosh(y_pred - y_true),axis=1)


def fourier_loss(y_true, y_pred):
    y_true_fft = tf.spectral.rfft(y_true)
    y_pred_fft = tf.spectral.rfft(y_pred)
    return tf.reduce_mean(tf.square(tf.abs(y_true_fft - y_pred_fft)),axis=1)

def robust_signal_decay_loss(y_true, y_pred):
    L = tf.shape(y_true)[1]  # 获取时间序列长度
    l_range = tf.range(1, L + 1, dtype=tf.float32)  # 生成从1到L的序列
    weights = tf.pow(l_range, -0.5)  # 计算权重 l^-1/2
    weights = tf.reshape(weights, [1, -1])  # 重塑为形状 (1, L)
    abs_errors = tf.abs(y_pred - y_true)  # 计算绝对误差
    weighted_errors = abs_errors * weights  # 乘以权重
    return tf.reduce_mean(weighted_errors,axis=1)  # 计算加权误差的平均值


def get_threshold():
    applianceThreshold=(params_appliance[applianceName]['on_power_threshold']-params_appliance[applianceName]['mean'])\
              /params_appliance[applianceName]['std']
    return applianceThreshold
applianceThreshold=get_threshold()
def switch_state_penalty(y_true, y_pred):
    true_state = tf.cast(tf.greater(y_true, applianceThreshold), tf.float32)
    pred_state = tf.cast(tf.greater(y_pred, applianceThreshold), tf.float32)
    print(y_true.shape,'--------------------------')
    penalty = tf.square(true_state - pred_state)
    return tf.reduce_mean(penalty,1)


# 综合损失函数
def combined_loss(y_true, y_pred, alpha=1, beta=0.1):
    # robust_loss = log_cosh_loss(y_true, y_pred)
    robust_loss=huber_loss(y_true, y_pred)
    switch_loss = switch_state_penalty(y_true, y_pred)
    # return alpha * robust_loss + beta * switch_loss
    # return alpha * tf.reduce_mean(robust_loss,1)  #当前效果最好
    return alpha * tf.reduce_mean(robust_loss, 1)+beta * switch_loss*switch
    # return alpha * tf.reduce_mean(tf.squared_difference(y, y_), 1) + beta * switch_loss


# 使用Log-Cosh损失函数
# cost = tf.reduce_mean(combined_loss(y_, y))
# cost = log_cosh_loss(y_, y)
# 使用Huber损失函数

if originModel:
    cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(y, y_), 1))
    # cost = tf.reduce_mean(combined_loss(y_, y))
else:
    cost = tf.reduce_mean(combined_loss(y_, y))
    # cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(y, y_), 1))



#cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(y, y_), 1))  # y_是预定义的，y含有预定义的x



# acc=get_accuracy(y_,y,50)

# model's weights to be trained  #####################################
train_params = tf.trainable_variables()
log("All network parameters: ")
log([v.name for v in train_params])
# if transfer learning is selected, just the dense layer will be trained
if not args.transfer_model and args.transfer_cnn:
    parameters = 10
else:
    parameters = 0
log("Trainable parameters:")
log([v.name for v in train_params[parameters:]])
#######################################################################
# Training hyper parameters
train_op = tf.train.AdamOptimizer(learning_rate=0.001,
                                  beta1=0.9,
                                  beta2=0.999,
                                  epsilon=1e-08,
                                  use_locking=False).minimize(cost,  # 含有预定义的x、y_
                                                              var_list=train_params[parameters:]
                                                             )
# train_op  = tf.train.GradientDescentOptimizer(learning_rate=0.001, momentum=0.9, decay=0.0005, nesterov=False)
# train_op = tf.compat.v1.train.RMSPropOptimizer(
#     learning_rate = 0.001, decay=0.0005, momentum=0.9, epsilon=1e-10, use_locking=False,
#     centered=False, name='RMSProp'
# )



###########################################
uninitialized_vars = []
for var in tf.all_variables():
    try:
        sess.run(var)
    except tf.errors.FailedPreconditionError:
        uninitialized_vars.append(var)

init_new_vars_op = tf.initialize_variables(uninitialized_vars)
sess.run(init_new_vars_op)
###########################################
log('TensorFlow Session starting...')

# TensorBoard summary (graph)
tf.summary.scalar('cost', cost)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./tensorboard_test')
writer.add_graph(sess.graph)
log('TensorBoard infos in ./tensorboard_test')
###############################################     Save path depending on the training behaviour
if not args.transfer_model and args.transfer_cnn:
    save_path = args.save_dir + '/easy1_' + appliance_name + '_transf_' + args.cnn + '_pointnet_model'
else:
    if originModel:
        save_path = args.save_dir + f'/easy{TrainNum}_{datasetName}' + appliance_name + f'{TrainPercent}_pointnet_model'
    else:
        # save_path = args.save_dir + f'/easy{TrainNum}_{datasetName}' + appliance_name + 'CombineEight5_pointnet_model'
        save_path = args.save_dir + f'/easy{TrainNum}_{datasetName}' + appliance_name + f'Combine{TrainPercent}_pointnet_model'

if not os.path.exists(save_path):
    os.makedirs(save_path)
################################################
# Calling custom training function
train_loss, val_loss, step_train_loss, step_val_loss = nf.customfit(sess=sess,
                                                                    network=model,  # 传入网络模型
                                                                    cost=cost,  # 传入cost function
                                                                    train_op=train_op,  # 训练参数（如学习率等
                                                                    train_provider=train_provider,  # 提供训练数据集
                                                                    x=x,
                                                                    y_=y_,
                                                                    acc=None,
                                                                    n_epoch=args.n_epoch,  # 100
                                                                    print_freq=1,
                                                                    val_provider=val_provider,
                                                                    save_model=args.save_model,
                                                                    save_path=save_path,
                                                                    epoch_identifier=None,
                                                                    earlystopping=True,
                                                                    min_epoch=1,
                                                                    patience=20)

# Following are training info

log('train loss: ' + str(train_loss))
log('val loss: ' + str(val_loss))
infos = pd.DataFrame(data={'train_loss': step_train_loss,
                           # 'val_loss': step_val_loss
                           })



# plt.figure
# epochs = range(1, len(train_loss) + 1)
# plt.plot(epochs, train_loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('S2S_CNN_loss-{}.png'.format(appliance_name))
# plt.show()
#------------------------
# infos.to_csv('./training_infos-{:}.csv'.format(appliance_name))
# # infos.to_csv('./training_infos-{:}-{:}-{:}.csv'.format(appliance_name, args.transfer, args.cnn))
# log('training infos in .csv file')

# This check that the CNN is the same of the beginning
# if not args.transfer_model and args.transfer_cnn:
#     log('Transfer learning check ...')
#     session = K.get_session()
#     for v in tf.trainable_variables():
#         if v.name == 'conv2d_1/kernel:0':
#             value = session.run(v)
#             vl = np.array(value).flatten()
#             c1 = np.array(cnn_check_weights).flatten()
#             if False in vl == c1:
#                 log('Transfer check --- ERROR ---')
#             else:
#                 log('Transfer check --- OK ---')
sess.close()
