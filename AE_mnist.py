##############################################################################
import argparse

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions  as F
from chainer import optimizers

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import sys, time, math


plt.style.use('ggplot')

##############################################################################

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
 # GPUが使えるか確認
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

##############################################################################

# draw a image of handwriting number
def draw_digit_ae(data, n, row, col, _type):
    size = 28
    plt.subplot(row, col, n)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,28)
    plt.ylim(0,28)
    plt.pcolor(Z)
    plt.title("type=%s"%(_type), size=8)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

##############################################################################

# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batchsize = 100

# 学習の繰り返し回数
#n_epoch   = 30
n_epoch   = 3

# 中間層の数
n_units   = 1000

# ノイズ付加有無
noised = False

# MNISTの手書き数字データのダウンロード
# #HOME/scikit_learn_data/mldata/mnist-original.mat にキャッシュされる
print('fetch MNIST dataset')
mnist = fetch_mldata('MNIST original', data_home="~/survey")
print('Complete!')

# mnist.data : 70,000件の784次元ベクトルデータ
mnist.data   = mnist.data.astype(np.float32)
mnist.data  /= 255     # 0-1のデータに変換

# mnist.target : 正解データ（教師データ）
mnist.target = mnist.target.astype(np.int32)

##############################################################################

# 学習用データを N個、検証用データを残りの個数と設定
N = 60000
y_train, y_test = np.split(mnist.data.copy(),   [N])
N_test = y_test.shape[0]

###
print('y_test :\n',y_test)
print('len(y_test) :', len(y_test))
print('len(y_test[0]) :', len(y_test[0]))
print('予想 784')
print('N_test :', N_test)
print('y_test.shape :', y_test.shape)
print('y_test.shape[0] :', y_test.shape[0])
print('y_test.shape[1] :', y_test.shape[1])
#print('y_test.shape[2] :', y_test.shape[2])
###

if noised:
    # Add noise
    noise_ratio = 0.2
    for data in mnist.data:
        perm = np.random.permutation(mnist.data.shape[1])[:int(mnist.data.shape[1]*noise_ratio)]
        data[perm] = 0.0
    
x_train, x_test = np.split(mnist.data,   [N])

# AutoEncoderのモデルの設定
# 入力 784次元、出力 784次元, 2層
model = chainer.FunctionSet(l1=F.Linear(784, n_units),
                            l2=F.Linear(n_units, 784))

# GPU使用の時はGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Neural net architecture
def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    y = F.dropout(F.relu(model.l1(x)),  train=train)
    x_hat  = F.dropout(model.l2(y),  train=train)
    # 誤差関数として二乗誤差関数を用いる
    return F.mean_squared_error(x_hat, t)

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

##############################################################################

l1_W = []
l2_W = []

l1_b = []
l2_b = []

train_loss = []
test_loss = []
test_mean_loss = []

prev_loss = -1
loss_std = 0

loss_rate = []

# Learning loop
for epoch in six.moves.range(1, n_epoch+1):
    print('epoch', epoch)
    start_time = time.clock()

    # training
    perm = np.random.permutation(N)
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch = xp.asarray(x_train[perm[i:i+batchsize]])
        y_batch = xp.asarray(y_train[perm[i:i+batchsize]])
        
        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        train_loss.append(loss.data)
        sum_loss     += float(loss.data) * batchsize
    print('\ttrain mean loss={} '.format(sum_loss / N))
    
    # evaluation
    sum_loss     = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = xp.asarray(x_test[i:i+batchsize])
        y_batch = xp.asarray(y_test[i:i+batchsize])
        loss = forward(x_batch, y_batch, train=False)

        test_loss.append(loss.data)
        sum_loss     += float(loss.data) * batchsize

    loss_val = sum_loss / N_test #batchsize(100)ごとのlossの合計(10000/100回足してる)/testデータの数(10000)
    print('\ttest  mean loss={}'.format(loss_val))

    if epoch == 1:
        loss_std = loss_val
        loss_rate.append(100)
    else:
        print('\tratio :%.3f'%(loss_val/loss_std * 100))
        loss_rate.append(loss_val/loss_std * 100)
        
    if prev_loss >= 0:
        diff = loss_val - prev_loss
        ratio = diff/prev_loss * 100
        print('\timpr rate:%.3f'%(-ratio))
    
    prev_loss = sum_loss / N_test
    test_mean_loss.append(loss_val)
    
    l1_W.append(model.l1.W)
    l2_W.append(model.l2.W)
    end_time = time.clock()
    print("\ttime = %.3f" %(end_time-start_time))

# Draw mean loss graph
plt.style.use('ggplot')
plt.figure(figsize=(10,7))
plt.plot(test_mean_loss, lw=1)
plt.title("")
plt.ylabel("mean loss")
plt.xlabel("epoch")
plt.savefig("./pic/AE_mnist/mean loss graph.png")

##############################################################################

###
print('start')
for i in range(len(l1_W[len(l1_W)-1])):
    print("len(l1_W[len(l1_W)-1][i]) : ",len(l1_W[len(l1_W)-1][i]))
print("len(l1_W[len(l1_W)-1]) : ",len(l1_W[len(l1_W)-1]))
print("len(l1_W)) : ",len(l1_W))
print('end')
###

##############################################################################

print('plot start : IO_images.png')
# 入力と出力を可視化
plt.style.use('fivethirtyeight')

#plt.figure(figsize=(15,25))
plt.figure(figsize=(10,20))

num = 100
cnt = 0
ans_list  = []
pred_list = []
for idx in np.random.permutation(N_test)[:num]:
    xxx = x_test[idx].astype(np.float32)
    h1 = F.dropout(F.relu(model.l1(chainer.Variable(xxx.reshape(1,784)))),  train=False)
    y  = model.l2(h1)
    cnt+=1
    ans_list.append(x_test[idx])
    pred_list.append(y)

cnt = 0
for i in range(int(num/10)):
    for j in range (10):
        img_no = i*10+j
        pos = (2*i)*10+j
        draw_digit_ae(ans_list[img_no],  pos+1, 20, 10, "ans")
        
    for j in range (10):
        img_no = i*10+j
        pos = (2*i+1)*10+j
        draw_digit_ae(pred_list[i*10+j].data, pos+1, 20, 10, "pred")
    
plt.savefig("./pic/AE_mnist/IO_images.png")
print('plot end : IO_images.png')

##############################################################################

print('plot start : w1_images.png')
# W(1)を可視化
plt.style.use('fivethirtyeight')
# draw digit images
def draw_digit_w1(data, n, i, length):
    size = 28
    plt.subplot(math.ceil(length/15), 15, n)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,size)
    plt.ylim(0,size)
    plt.pcolor(Z)
    plt.title("%d"%i, size=9)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

#plt.figure(figsize=(15,67))
plt.figure(figsize=(15,55))
cnt = 1

for i in range(len(l1_W[len(l1_W)-1])):
    draw_digit_w1(l1_W[len(l1_W)-1][i], cnt, i, len(l1_W[len(l1_W)-1]))
    cnt += 1
    
plt.savefig("./pic/AE_mnist/w1_images.png")
print('plot end : w1_images.png')

##############################################################################

print('plot start : w2_T_images.png')
# W(2).Tを可視化
plt.style.use('fivethirtyeight')
# draw digit images

def draw_digit2(data, i, length):
    size = 28
    plt.subplot(math.ceil(length/15)+1, 15, i+1)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(Z)
    plt.title("%d"%i, size=9)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

W_T = np.array(l2_W[len(l2_W)-1]).T

#plt.figure(figsize=(15,67))
plt.figure(figsize=(15,55))
for i in range(W_T.shape[0]):
    draw_digit2(W_T[i], i, W_T.shape[0])
    
plt.savefig("./pic/AE_mnist/w2_T_images.png")
print('plot end : w2_T_images.png')

##############################################################################
