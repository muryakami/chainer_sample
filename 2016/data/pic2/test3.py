##############################################################################
import argparse

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import time
import math
import pickle

##############################################################################

workspace = '/Users/yuki/survey/workspace/'
fig_home = '{}2016/data/pic2/outline/26/'.format(workspace)

##############################################################################

print('fetch MNIST dataset')
mnist = fetch_mldata('MNIST original', data_home='~/survey/')
print('Complete!')
mnist.data = mnist.data.astype(np.float32) / 255
train_data = mnist.data[:60000]
test_data = mnist.data[60000:]

##############################################################################

depth = 2
order = (('enc1', 'enc2'), ('dec2', 'dec1'))

model = pickle.load(open('{}2016/data/pic2/26/test_SAE.pkl'.format(workspace), 'rb'))

##############################################################################

# draw digit images
def draw_layer(data, index, row, column):
    size = math.sqrt(data.shape[0])
    plt.subplot(row, column, index+1)  # 行数, 列数, プロット番号
    Z = data.reshape(size, size)       # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                     # flip vertical
    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.pcolor(Z)
    plt.title('{}'.format(index), size=8)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')

##############################################################################

def image_save(data, str):
    save_home = '{}{}.png'.format(fig_home, str)
    print('plot start : {}.png'.format(str))
    
    plt.style.use('fivethirtyeight')

    column = 5
    row = math.ceil(data.shape[0]/column)
    plt.figure(figsize=(column, row))

    for i in six.moves.range(data.shape[0]):
        draw_layer(data[i], i, row, column)
        
    plt.savefig(save_home, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    print('plot end : {}.png'.format(str))
    print()

##############################################################################

# Neural net architecture
# 途中からデータを流すニューラルネットの構造
def enc_forward(x_data, layer):
    x = chainer.Variable(x_data.astype(np.float32))
    y = model.enc1(x)
    if layer <= 1:
        return y
    y = model.enc2(F.sigmoid(y))
    return y

##############################################################################

for i in six.moves.range(1, depth+1):
    enc_hidden_x = np.identity(28**2)
    enc_hidden_y = enc_forward(enc_hidden_x, i)
    enc_hidden_T = np.array(enc_hidden_y.data).T
    image_save(enc_hidden_T, 'enc_hL{}_images'.format(i))

##############################################################################

# draw a image of handwriting number
def draw_digit_io(data, index, row, column, _type):
    size = 28
    plt.subplot(row, column, index+1)
    Z = data.reshape(size, size)
    Z = Z[::-1,:]
    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.pcolor(Z)
    plt.title('{}'.format(_type), size=8)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')


print('plot start : IO_images.png')
# 入力と出力を可視化
num = 5
column = 5
row = 1
ans_list  = []
pred_list = []

plt.style.use('fivethirtyeight')
plt.figure(figsize=(column, row*2))

for i in np.random.permutation(len(test_data))[:num]:
    sample = test_data[i].astype(np.float32)
    y1 = model.enc1(chainer.Variable(sample.reshape(1,784)))
    y2 = model.enc2(F.sigmoid(y1))
    y3 = model.dec2(F.sigmoid(y2))
    y  = model.dec1(F.sigmoid(y3))
    ans_list.append(test_data[i])
    pred_list.append(y.data)

for i in range(column):
    ans_pos = i
    pred_pos = column + i
    draw_digit_io(ans_list[i],  ans_pos, row*2, column, 'IN')
    draw_digit_io(pred_list[i], pred_pos, row*2, column, 'OUT')

plt.savefig('{}IO_images.png'.format(fig_home), bbox_inches='tight', pad_inches=0.0)
plt.close()
print('plot end : IO_images.png')
print()

##############################################################################
