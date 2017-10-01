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
fig_home = '{}2016/data/pic/'.format(workspace)

##############################################################################

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('model') ###
 # GPUが使えるか確認
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

##############################################################################

# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batchsize = 100
# 学習の繰り返し回数
n_epoch   = 500
# 中間層の数
n_units   = (28**2, 4**2, 10)

##############################################################################

print('fetch MNIST dataset')
mnist = fetch_mldata('MNIST original', data_home='~/survey/')
print('Complete!')
perm = np.random.permutation(len(mnist.data))
mnist.data = mnist.data.astype(np.float32) / 255
train_data = mnist.data[perm][:60000]
test_data = mnist.data[perm][60000:]

print(perm)
print(len(perm))
print(len(mnist.data))
print(mnist.data)
print(len(mnist.data[perm]))
print(mnist.data[perm])
print(len(mnist.data[0]))
print(mnist.data[0])


#test = [i for i in range(10)]
#for i in range(10):
#    for j in range(10):
#        (i, j)
test = [[i, j] for i in range(10) for j in range(10)]
tperm = np.random.permutation(len(test))
print(test)
print(tperm)
#print(test[tperm])
nptest = np.asarray(test)
print(nptest[tperm])

print(mnist.data[[0,1]])

print(nptest[[15,40]])
