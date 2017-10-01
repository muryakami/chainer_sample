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
import pickle

##############################################################################

workspace = '/Users/yuki/survey/workspace/'
fig_home = '{}pic_AE_Linear/'.format(workspace)

##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('model')
 # GPUが使えるか確認
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

##############################################################################

# draw a image of handwriting number
def draw_digit_ae(data, index, row, column, _type):
    size = 28
    plt.subplot(row, column, index+1)  # 行数, 列数, プロット番号
    Z = data.reshape(size, size)       # convert from vector to 28x28 matrix
    Z = Z[::-1, :]                     # flip vertical
    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.pcolor(Z)
    plt.title('type=%s'%(_type), size=8)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')

# draw digit images
def draw_layer(data, index, length):
    column = 15
    row = math.ceil(length/column)
    size = 28
    plt.subplot(row, column, index+1)  # 行数, 列数, プロット番号
    Z = data.reshape(size, size)       # convert from vector to 28x28 matrix
    Z = Z[::-1, :]                     # flip vertical
    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.pcolor(Z)
    plt.title('%d'%index, size=8)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')

# draw(data, index, length, column, str)で文字列としてstrを渡すのが良い？

##############################################################################

def image_save(data, str):
    save_home = '{}{}.png'.format(fig_home, str)
    print('plot start : {}.png'.format(str))
    
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(15,70))
    
    print('data.shape :', data.shape)
    print('data.shape[0] :', data.shape[0])

    for i in six.moves.range(data.shape[0]):
        draw_layer(data[i], i, data.shape[0])
        
    plt.savefig(save_home, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    print('plot end : {}.png'.format(str))
    print()

##############################################################################

# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batchsize = 100

# 学習の繰り返し回数
n_epoch   = 1000

# 層の数
#n_units   = (28**2, 10**2)
n_units   = (32**2, 10**2)

"""
# MNISTの手書き数字データのダウンロード
print('fetch MNIST dataset')
mnist = fetch_mldata('MNIST original', data_home='~/survey')
print('Complete!')

# mnist.data : 70,000件の784次元ベクトルデータ
mnist.data   = mnist.data.astype(np.float32)
mnist.data  /= 255     # 0-1のデータに変換

# mnist.target : 正解データ（教師データ）
mnist.target = mnist.target.astype(np.int32)
"""

def unpickle(f):
    import pickle
    fo = open(f, 'rb')
    d = pickle.load(fo, encoding='latin1')
    fo.close()
    return d

def load_cifar10(datadir):
    train_data = []
    train_target = []

    # 訓練データをロード
    for i in range(1, 6):
        d = unpickle('%s/data_batch_%d' % (datadir, i))
        train_data.extend(d['data'])
        train_target.extend(d['labels'])

    # テストデータをロード
    d = unpickle('%s/test_batch' % (datadir))
    test_data = d['data']
    test_target = d['labels']

    # データはfloat32、ラベルはint32のndarrayに変換
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.int32)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.int32)

    # 画像のピクセル値を0-1に正規化
    train_data /= 255.0
    test_data /= 255.0

    return train_data, test_data, train_target, test_target

##############################################################################

"""
# 学習用データを N個、検証用データを残りの個数と設定
N = 60000
x_train, x_test = np.split(mnist.data, [N])
y_train, y_test = np.split(mnist.data.copy(), [N])
N_test = y_test.shape[0]
"""

# CIFAR-10データをロード
print('load CIFAR-10 dataset')
X_train, X_test, y_train, y_test = load_cifar10('~/survey/aidiary/cifar-10')

N = y_train.size
N_test = y_test.size

# 画像を (nsample, channel, height, width) の4次元テンソルに変換
X_train = X_train.reshape((len(X_train), 3, 32, 32))
X_test = X_test.reshape((len(X_test), 3, 32, 32))

# AutoEncoderのモデルの設定
# 入力 784次元、出力 784次元, 2層
if args.model == '':
    model = chainer.FunctionSet(l1=F.Linear(n_units[0], n_units[1]),
                                l2=F.Linear(n_units[1], n_units[0]))
else:
    model = pickle.load(open(args.model, 'rb'))

# GPU使用の時はGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Neural net architecture
"""
def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    y = F.dropout(F.relu(model.l1(x)),  train=train)
    x_hat  = F.dropout(model.l2(y),  train=train)
    #print('x_hat.data :\n',x_hat.data) ###
    #print('x_hat.data.shape :', x_hat.data.shape) ###
    # 誤差関数として二乗誤差関数を用いる
    return F.mean_squared_error(x_hat, t)
"""
# テスト
def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    y = model.l1(x)
    x_hat  = model.l2(y)
    #y = F.Linear(model.l1(x))
    #x_hat  = F.Linear(model.l2(y))
    # 誤差関数として二乗誤差関数を用いる
    return F.mean_squared_error(x_hat, t)

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

##############################################################################

def dec_forward(x_data):
    x = chainer.Variable(x_data.astype(np.float32))
    y = model.l2(x)
    return y

def enc_forward(x_data):
    x = chainer.Variable(x_data.astype(np.float32))
    y = model.l1(x)
    return y

test_mean_loss = []

fp1 = open('accuracy.txt', 'w')
fp2 = open('loss.txt', 'w')
fp1.write('epoch\ttest_accuracy\n')
fp2.write('epoch\ttrain_loss\n')

period = 100
# Learning loop
for epoch in six.moves.range(1, int(n_epoch/period)+1):

    for p in six.moves.range(period):
        print('epoch : {}'.format((epoch-1)*period +(p+1)))
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
            
            sum_loss += float(loss.data) * len(y_batch)
            
        print('\t train mean loss : %f' % (sum_loss / N))
        fp2.write('%d\t%f\n' % (epoch, sum_loss / N))
        fp2.flush()
        
        # evaluation
        sum_accuracy = 0
        for i in six.moves.range(0, N_test, batchsize):
            x_batch = xp.asarray(x_test[i:i+batchsize])
            y_batch = xp.asarray(y_test[i:i+batchsize])
            
            acc = forward(x_batch, y_batch, train=False)
            
            sum_accuracy += float(acc.data) * len(y_batch)
            
        loss_val = sum_accuracy / N_test

        print('\t test accuracy : %f' % (sum_accuracy / N_test))
        fp1.write('%d\t%f\n' % (epoch, sum_accuracy / N_test))
        fp1.flush()
        
        test_mean_loss.append(loss_val)

        end_time = time.clock()
        print('\t time : %.3f' %(end_time-start_time))

        fp1.close()
        fp2.close()

    # 可視化メソッド
    ew1 = np.array(model.l1.W)
    image_save(ew1, 'ew1({})_images'.format(epoch * period))
    dw1_T = np.array(model.l2.W).T
    image_save(dw1_T, 'dw1_T({})_images'.format(epoch * period))
    # ......
    dec_hidden_x = np.identity(n_units[1])
    dec_hidden_y = dec_forward(dec_hidden_x)
    dec_hidden = np.array(dec_hidden_y.data)
    image_save(dec_hidden, 'dec_hL1({})_images'.format(epoch * period))
    enc_hidden_x = np.identity(n_units[0])
    enc_hidden_y = enc_forward(enc_hidden_x)
    enc_hidden_T = np.array(enc_hidden_y.data).T
    image_save(enc_hidden_T, 'enc_hL1_T({})_images'.format(epoch * period))

    # 消滅する勾配問題検証
    test_ew1 = chainer.Variable(ew1)
    test_dw1_T = chainer.Variable(dw1_T)
    print('Layer :', F.mean_squared_error(test_ew1, test_dw1_T).data)
    # 画像一致率（二乗誤差）
    test_enc_hidden_T = chainer.Variable(enc_hidden_T)
    print('画像一致率（二乗誤差）')
    print('ew1_images, enc_hidden_T : ', F.mean_squared_error(test_ew1, test_enc_hidden_T).data)
    # 平均二乗誤差平方根
    print('RMSE :', math.sqrt(((ew1 - enc_hidden_T)**2).mean()))

model.to_cpu()
pickle.dump(model, open('{}model.pkl'.format(fig_home), 'wb'), -1)

# Draw mean loss graph
plt.style.use('ggplot')
plt.figure(figsize=(10,7))
plt.plot(test_mean_loss, lw=1)
plt.title('mean loss graph')
plt.ylabel('mean loss')
plt.xlabel('epoch')
plt.savefig('{}mean loss graph.png'.format(fig_home))
plt.close()

##############################################################################

print('plot start : IO_images.png')
# 入力と出力を可視化
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,25))

num = 100
column = 10
row = int(num/column) # 割り切れる必要あり
ans_list  = []
pred_list = []
for idx in np.random.permutation(N_test)[:num]:
    xxx = x_test[idx].astype(np.float32)
    # 評価なので dropout はしない
    h1 = F.relu(model.l1(chainer.Variable(xxx.reshape(1,n_units[0]))))
    y  = model.l2(h1)
    # h1 = F.dropout(F.relu(model.l1(chainer.Variable(xxx.reshape(1,784)))),  train=False)
    # y  = F.dropout(model.l2(h1), train=False)
    # と同義
    ans_list.append(x_test[idx])
    pred_list.append(y)

for i in six.moves.range(row):
    for j in six.moves.range(column):
        img_no = i *row +j
        ans_pos = (2*i) *row +j
        pred_pos = (2*i+1) *row +j
        draw_digit_ae(ans_list[img_no],  ans_pos, row*2, column, 'ans')
        draw_digit_ae(pred_list[img_no].data, pred_pos, row*2, column, 'pred')

plt.savefig('{}IO_images.png'.format(fig_home), bbox_inches='tight', pad_inches=0.0)
plt.close()
print('plot end : IO_images.png')
print()

##############################################################################
