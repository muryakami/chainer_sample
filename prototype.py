##############################################################################

#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.

"""
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

import data
import time ###


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
 # GPUが使えるか確認
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

 # 学習のパラメータ
batchsize = 100 # 確率的勾配降下法で学習させる際の1回のバッチサイズ
n_epoch = 20 # 学習の繰り返し回数
n_units = 1000 # 中間層の数

# Prepare dataset データセットをロード
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

# 訓練データとテストデータに分割
N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

##############################################################################

# Prepare multi-layer perceptron model 多層パーセプトロンのモデル(パラメータ集合)
model = chainer.FunctionSet(l1=F.Linear(784, n_units),
                            l2=F.Linear(n_units, n_units),
                            l3=F.Linear(n_units, 10))
# GPU使用の時はGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


def forward(x_data, y_data, train=True):
    # Neural net architecture 順伝播の処理を定義
    # 入力と教師データ
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    # 隠れ層1の出力
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    # 隠れ層2の出力
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    # 出力層の出力
    y = model.l3(h2)
    # 損失と精度 損失は多値分類なのでクロスエントロピーを使う
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


# Setup optimizer Optimizerをセット
# 最適化対象であるパラメータ集合のmodelを渡しておく
optimizer = optimizers.Adam()
# optimizer = optimizers.AdaDelta()
# optimizer = optimizers.AdaGrad()
# optimizer = optimizers.MomentumSGD()
# optimizer = optimizers.RMSprop()
# optimizer = optimizers.SGD()
optimizer.setup(model)

##############################################################################

# Learning loop 訓練ループ
# 各エポックでテスト精度を求める
start_time = time.clock() ###
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training 訓練データを用いてパラメータを更新する
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch = xp.asarray(x_train[perm[i:i + batchsize]])
        y_batch = xp.asarray(y_train[perm[i:i + batchsize]])

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        if epoch == 1 and i == 0:
            with open("graph.dot", "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open("graph.wo_split.dot", "w") as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

##############################################################################

    # evaluation 訓練データを用いて精度を評価する
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = xp.asarray(x_test[i:i + batchsize])
        y_batch = xp.asarray(y_test[i:i + batchsize])

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

end_time = time.clock() ###
print(end_time - start_time) ###
