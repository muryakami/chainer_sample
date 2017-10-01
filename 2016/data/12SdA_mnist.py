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
fig_home = '{}2016/data/pic/12/'.format(workspace)

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
n_units   = (28**2, 12**2, 10)

##############################################################################

print('fetch MNIST dataset')
mnist = fetch_mldata('MNIST original', data_home='~/survey/')
print('Complete!')
perm = np.random.permutation(len(mnist.data))
mnist.data = mnist.data.astype(np.float32) / 255
train_data = mnist.data[perm][:60000]
test_data = mnist.data[perm][60000:]

##############################################################################

depth = 2 ###
order = (('enc1', 'enc2'), ('dec2', 'dec1')) ###

#model = chainer.FunctionSet(
#    enc1=F.Linear(n_units[0], n_units[1]),
#    enc2=F.Linear(n_units[1], n_units[2]),
#    dec2=F.Linear(n_units[2], n_units[1]),
#    dec1=F.Linear(n_units[1], n_units[0])
#)
if args.model == '': ###
    model = chainer.FunctionSet(
        enc1=F.Linear(n_units[0], n_units[1]),
        enc2=F.Linear(n_units[1], n_units[2]),
        dec2=F.Linear(n_units[2], n_units[1]),
        dec1=F.Linear(n_units[1], n_units[0])
    )
else: ###
    model = pickle.load(open(args.model, 'rb')) ###

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

    #depth = len(encl)

    for (el, dl) in zip(encl, reversed(decl)):
        submodel = chainer.FunctionSet(
            enc=model[el],
            dec=model[dl]
        )
        sublayer.append(submodel)


set_order(*order)

##############################################################################

# optimizer = optimizers.AdaDelta()
optimizer = optimizers.Adam()
optimizer.setup(model)

##############################################################################

def __encode(x, layer, train):
    if train:
        x = F.dropout(x, ratio=0.2, train=train)
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
                #x_batch = xp.asarray(x_data[perm[i:i+batchsize]])

                optimizer.zero_grads()
                err = validate(x_batch, layer=l, train=True)

                err.backward()
                optimizer.update()

                sum_error += float(chainer.cuda.to_cpu(err.data)) * len(x_batch)
                #sum_error += float(err.data) * len(x_batch)
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
            #x_batch = xp.asarray(x_data[perm[i:i+batchsize]])
            y = forward(x_batch, train=False)

            if args.gpu >= 0:
                x_batch = chainer.cuda.to_gpu(x_batch)
            x = chainer.Variable(x_batch)

            err = loss_function(x, y, **loss_param)

            sum_error += float(chainer.cuda.to_cpu(err.data)) * len(x_batch)
            action()

        return sum_error / N

##############################################################################

# save trained network parameters to file
def save_param(dst='./network.param.npy'):
    # model.to_cpu() seems to change itself
    # This causes step-by-step saving each epochs with gpu
    param = np.array(model.to_cpu().parameters)
    np.save(dst, param)
    if args.gpu >= 0:
        model.to_gpu()

# load pre-trained network parameters from file
def load_param(src='./network.param.npy'):
    if not os.path.exists(src):
        raise IOError('specified parameter file does not exists')

    param = np.load(src)
    model.copy_parameters_from(param)

    # by this process, model parameters to be cpu_array
    if args.gpu >= 0:
        model = model.to_gpu()

##############################################################################

# draw digit images
#def draw_layer(data, index, length):
def draw_layer(data, index, row, column):
    #column = math.sqrt(length)
    #row = math.sqrt(length)
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
    
    print('data.shape :', data.shape)
    print('data.shape[0] :', data.shape[0])

    for i in six.moves.range(data.shape[0]):
        # draw_layer(data[i], i, data.shape[0])
        draw_layer(data[i], i, row, column)
        
    plt.savefig(save_home, bbox_inches='tight', pad_inches=0.0)
    plt.close() #
    print('plot end : {}.png'.format(str))
    print()

##############################################################################

# Neural net architecture
# 途中からデータを流すニューラルネットの構造
def dec_forward(x_data, layer):
    x = chainer.Variable(x_data.astype(np.float32))
    if layer >= 2:
        x = F.sigmoid(model.dec2(x))
    y = model.dec1(x)
    return y

def enc_forward(x_data, layer):
    x = chainer.Variable(x_data.astype(np.float32))
    y = model.enc1(x)
    if layer <= 1:
        return y
    y = model.enc2(F.sigmoid(y))
    return y

##############################################################################

# 手書き数字データを描画する関数
def draw_dot(data, str):
    save_home = '{}{}.png'.format(fig_home, str)
    print('plot start : {}.png'.format(str))

    plt.style.use('fivethirtyeight')

    size = 784
    plt.figure(figsize=(size/10, size/10*1.045))

    print('data.shape :', data.shape)

    X, Y = np.meshgrid(range(size), range(size))
    Z = data.reshape(size, size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

    #plt.savefig(save_home, bbox_inches='tight', pad_inches=0.0)
    plt.savefig(save_home)
    plt.close()
    print('plot end : {}.png'.format(str))

##############################################################################

train_err = []
test_err = []

#period = 1000
period = 100
#period = 10
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

        save_param('{}sda.param.npy'.format(fig_home))

    # 可視化メソッド
    # ew1 = np.array(model.enc1.W)
    # image_save(ew1, 'ew1({})_images'.format(epoch * period))
    # ew2 = np.array(model.enc2.W)
    # image_save(ew2, 'ew2({})_images'.format(epoch * period))
    # dw2_T = np.array(model.dec2.W).T
    # image_save(dw2_T, 'dw2_T({})_images'.format(epoch * period))
    # dw1_T = np.array(model.dec1.W).T
    # image_save(dw1_T, 'dw1_T({})_images'.format(epoch * period))
    # el2 = np.dot(ew2, ew1)
    # image_save(el2, 'el2({})_images'.format(epoch * period))
    # dl2 = np.dot(dw2_T, dw1_T)
    # image_save(dl2, 'dl2({})_images'.format(epoch * period))
    # draw_dot(np.dot(el2.T, dl2), 'dot_raw({})_images'.format(epoch * period))
    #......
    for i in six.moves.range(1, depth+1):
        # identity: 単位行列の生成(正方行列), eye: identityと似ているが列数指定出来る
        # dec_hidden_x = np.identity(n_units[i])
        # dec_hidden_y = dec_forward(dec_hidden_x, i)
        # hidden layerを可視化
        # dec_hidden = np.array(dec_hidden_y.data)
        # image_save(dec_hidden, 'dec_hL{}({})_images'.format(i, epoch * period))
        
        # identity: 単位行列の生成(正方行列), eye: identityと似ているが列数指定出来る
        enc_hidden_x = np.identity(n_units[0])
        enc_hidden_y = enc_forward(enc_hidden_x, i)
        # hidden layerを可視化
        enc_hidden_T = np.array(enc_hidden_y.data).T
        image_save(enc_hidden_T, 'enc_hL{}({})_T_images'.format(i, epoch * period))

        # draw_dot(np.dot(dec_hidden.T, enc_hidden_T), 'dot_func{}({})_images'.format(i, epoch * period))

    # 消滅する勾配問題検証
    # test_ew1 = chainer.Variable(ew1)
    # test_dw1_T = chainer.Variable(dw1_T)
    # print('Layer1 :', F.mean_squared_error(test_ew1, test_dw1_T).data)
    # test_ew2 = chainer.Variable(ew2)
    # test_dw2_T = chainer.Variable(dw2_T)
    # print('Layer2 :', F.mean_squared_error(test_ew2, test_dw2_T).data)

# 精度と誤差をグラフ描画
plt.style.use('ggplot')
plt.figure(figsize=(8,6))
# plt.plot(range(len(train_err)), train_err)
plt.plot(range(len(test_err)), test_err)
plt.legend(['test_err'], loc='upper right')
# plt.legend(['train_err','test_err'], loc=4)
plt.title('Accuracy of digit recognition.')
plt.plot()
plt.savefig('{}err.png'.format(fig_home), bbox_inches='tight', pad_inches=0.0)
plt.close()

model.to_cpu()
pickle.dump(model, open('{}test_SAE.pkl'.format(fig_home), 'wb'), -1)

##############################################################################
