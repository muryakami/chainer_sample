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
fig_home = '{}2016/data/pic2/26/'.format(workspace)

##############################################################################

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('model')
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
n_units   = (28**2, 26**2, 10)

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

if args.model == '':
    model = chainer.FunctionSet(
        enc1=F.Linear(n_units[0], n_units[1]),
        enc2=F.Linear(n_units[1], n_units[2]),
        dec2=F.Linear(n_units[2], n_units[1]),
        dec1=F.Linear(n_units[1], n_units[0])
    )
else:
    model = pickle.load(open(args.model, 'rb'))

##############################################################################

# GPU使用の時はGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

##############################################################################

sublayer = []

loss_function = F.mean_squared_error
loss_param = {}

##############################################################################

def set_order(encl, decl):
    if len(encl) != len(decl):
        raise TypeError('Encode/Decode layers mismatch')

    for (el, dl) in zip(encl, reversed(decl)):
        submodel = chainer.FunctionSet(
            enc=model[el],
            dec=model[dl]
        )
        sublayer.append(submodel)

##############################################################################

set_order(*order)

optimizer = optimizers.Adam()
optimizer.setup(model)

##############################################################################

def __encode(x, layer, train):
    if layer == 0:
        return x
    x = F.sigmoid(model.enc1(x))
    if layer == 1:
        return x
    x = F.sigmoid(model.enc2(x))
    if layer == 2:
        return x
    return x

##############################################################################

def __decode(x, layer=None, train=False):
    if not train or layer == 2:
        x = F.sigmoid(model.dec2(x))
    if not train or layer == 1:
        x = F.sigmoid(model.dec1(x))
    return x

##############################################################################

def encode(x_data, layer=None, train=False):
    if args.gpu >= 0:
        x_data = chainer.cuda.to_gpu(x_data)

    x = chainer.Variable(x_data)

    return __encode(x, layer, train)

##############################################################################

def validate(x_data, layer=None, train=False):
    targ = encode(x_data, layer-1, train=False)
    code = encode(x_data, layer, train=train)
    
    y = __decode(code, layer, train=train)
    
    return loss_function(targ, y, **loss_param)


def train(x_data, batchsize=100, action=(lambda: None)):
        errs = []
        N = len(x_data)
        perm = np.random.permutation(N)

        for l in range(1, depth+1):
            optimizer.setup(sublayer[l-1])

            sum_error = 0.

            for i in range(0, N, batchsize):
                x_batch = x_data[perm[i:i+batchsize]]

                optimizer.zero_grads()
                err = validate(x_batch, layer=l, train=True)

                err.backward()
                optimizer.update()

                sum_error += float(chainer.cuda.to_cpu(err.data)) * len(x_batch)

                action()

            errs.append(sum_error / N)

        return tuple(errs)

##############################################################################

def forward(x_data, train=False):
    code = encode(x_data, train=train)
    y = __decode(code, train=train)
    
    return y


def test(x_data, batchsize=100, action=(lambda: None)):
        N = len(x_data)
        perm = np.random.permutation(N)

        sum_error = 0.

        for i in range(0, N, batchsize):
            x_batch = x_data[perm[i:i+batchsize]]
            y = forward(x_batch, train=False)

            if args.gpu >= 0:
                x_batch = chainer.cuda.to_gpu(x_batch)
            x = chainer.Variable(x_batch)

            err = loss_function(x, y, **loss_param)

            sum_error += float(chainer.cuda.to_cpu(err.data)) * len(x_batch)
            action()

        return sum_error / N

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
    plt.title('%d'%index, size=8)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')

##############################################################################

def image_save(data, str):
    save_home = '{}{}.png'.format(fig_home, str)
    print('plot start : {}.png'.format(str))
    
    plt.style.use('fivethirtyeight')

    column = 10
    row = math.ceil(data.shape[0]/column)
    plt.figure(figsize=(column, row*1.045))
    
    for i in six.moves.range(data.shape[0]):
        draw_layer(data[i], i, row, column)
        
    plt.savefig(save_home, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    print('plot end : {}.png'.format(str))
    print()

##############################################################################

# 途中までデータを流すニューラルネットの構造
def enc_forward(x_data, layer):
    x = chainer.Variable(x_data.astype(np.float32))
    y = model.enc1(x)
    if layer <= 1:
        return y
    y = model.enc2(F.sigmoid(y))
    return y

##############################################################################

train_err = []
test_err = []

period = 100

for epoch in six.moves.range(1, int(n_epoch/period)+1):

    for i in six.moves.range(period):
        print('epoch : %d' %((epoch-1)*period +(i+1)))

        err = train(train_data, batchsize=200)
        print(err)
        train_err.append(err)

        perm = np.random.permutation(len(test_data))
        terr = test(test_data[perm][:100])
        print(terr)
        test_err.append(terr)

        with open('{}sda.log'.format(fig_home), mode='a') as f:
            f.write("%d %f %f %f\n" % ((epoch-1)*period +(i+1), err[0], err[1], terr))

    for i in six.moves.range(1, depth+1):
        enc_hidden_x = np.identity(n_units[0])
        enc_hidden_y = enc_forward(enc_hidden_x, i)
        enc_hidden_T = np.array(enc_hidden_y.data).T
        image_save(enc_hidden_T, 'enc_hL{}({})_T_images'.format(i, epoch * period))

model.to_cpu()
pickle.dump(model, open('{}test_SAE.pkl'.format(fig_home), 'wb'), -1)

##############################################################################
