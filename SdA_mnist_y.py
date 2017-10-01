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

#from libdnn import StackedAutoEncoder

##############################################################################

workspace = '/Users/yuki/survey/workspace/'
fig_home = '{}pic_test5/'.format(workspace)

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
n_epoch   = 30
# 中間層の数
#n_units   = (28**2, 16**2, 8**2, 10)
n_units   = (28**2, 14**2, 7**2)
# ノイズ付加有無
#noised = False

##############################################################################
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


# 学習用データを N個、検証用データを残りの個数と設定
N = 60000
y_train, y_test = np.split(mnist.data.copy(), [N])
#N_test = y_test.shape[0]
N_test = y_test.size

#if noised:
#    # Add noise
#    noise_ratio = 0.2
#    for data in mnist.data:
#        perm = np.random.permutation(mnist.data.shape[1])[:int(mnist.data.shape[1]*noise_ratio)]
#        data[perm] = 0.0

x_train, x_test = np.split(mnist.data, [N])
"""

###
print('fetch MNIST dataset')
mnist = fetch_mldata('MNIST original', data_home='~/survey/')
print('Complete!')
perm = np.random.permutation(len(mnist.data))
mnist.data = mnist.data.astype(np.float32) / 255
train_data = mnist.data[perm][:60000]
test_data = mnist.data[perm][60000:]
###

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
"""
def encode(self, x, layer, train):
    if train:
        x = F.dropout(x, ratio=0.2, train=train)
    if layer == 0:
        return x
    x = F.sigmoid(self.model.enc1(x))
    if layer == 1:
        return x
    x = F.sigmoid(self.model.enc2(x))
    if layer == 2:
        return x
    return x

def decode(self, x, layer=None, train=False):
    if not train or layer == 2:
        x = F.sigmoid(self.model.dec2(x))
    if not train or layer == 1:
        x = F.sigmoid(self.model.dec1(x))
    return x
"""
##############################################################################

# GPU使用の時はGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

##############################################################################
"""
sda = StackedAutoEncoder(model, gpu=0)
sda.set_order(('enc1', 'enc2'), ('dec2', 'dec1'))
sda.set_optimizer(optimizers.AdaDelta)
sda.set_encode(encode)
sda.set_decode(decode)
"""
##############################################################################
"""
# Learning loop
for epoch in six.moves.range(1, n_epoch+1):
    print('epoch : %d' % (epoch))

    err = sda.train(train_data, batchsize=200)
    print(err)

    perm = np.random.permutation(len(test_data))
    terr = sda.test(test_data[perm][:100])
    print(terr)

    with open('sda.log', mode='a') as f:
        f.write("%d %f %f %f\n" % (epoch, err[0], err[1], terr))

    sda.save_param('sda.param.npy')
"""
##############################################################################
"""
def train(self, x_data, batchsize=100, action=(lambda: None)):
        errs = []
        N = len(x_data)
        perm = np.random.permutation(N)

        for l in range(1, self.depth + 1):
            self.optimizer.setup(self.sublayer[l - 1])

            sum_error = 0.

            for i in range(0, N, batchsize):
                x_batch = x_data[perm[i:i + batchsize]]

                self.optimizer.zero_grads()
                err = self.validate(x_batch, layer=l, train=True)

                err.backward()
                self.optimizer.update()

                sum_error += float(chainer.cuda.to_cpu(err.data)) * len(x_batch)
                #sum_error += float(err.data) * len(x_batch)
                action()

            errs.append(sum_error / N)

        return tuple(errs)

def test(self, x_data, batchsize=100, action=(lambda: None)):
        N = len(x_data)
        perm = np.random.permutation(N)

        sum_error = 0.

        for i in range(0, N, batchsize):
            x_batch = x_data[perm[i:i + batchsize]]
            y = self.forward(x_batch, train=False)

            if self.gpu >= 0:
                x_batch = chainer.cuda.to_gpu(x_batch)
            x = chainer.Variable(x_batch)

            err = self.loss_function(x, y, **self.loss_param)

            sum_error += float(chainer.cuda.to_cpu(err.data)) * len(x_batch)
            action()

        return sum_error / N

# こっちのvalidateが正解
def validate(self, x_data, layer=None, train=False):
    targ = self.encode(x_data, layer - 1, train=False)
    code = self.encode(x_data, layer, train=train)
    
    y = self.__decode(code, layer, train=train)
    
    return self.loss_function(targ, y, **self.loss_param)
""""""
encode と __encode,
decode と __decode
の違い
set_encode, set_decode しているのなら同じではないのか？
　→ decodeはない．前処理を行いたい場合にencodeを呼んでいる
""""""

def forward(self, x_data, train=False):
    code = self.encode(x_data, train=train)
    y = self.__decode(code, train=train)
    
    return y

def encode(self, x_data, layer=None, train=False):
    if self.gpu >= 0:
        x_data = chainer.cuda.to_gpu(x_data)

    x = chainer.Variable(x_data)

    return self.__encode(x, layer, train)

def __encode(self, x, layer, train):
    pass

def __decode(self, x, layer, train):
    pass


def set_order(self, encl, decl):
    if len(encl) != len(decl):
        raise TypeError('Encode/Decode layers mismatch')

    self.depth = len(encl)

    for (el, dl) in zip(encl, reversed(decl)):
        self.sublayer.append(chainer.FunctionSet(enc=self.model[el], dec=self.model[dl]))

def set_encode(self, func):
    self.__encode = MethodType(func, self, StackedAutoEncoder)

def set_decode(self, func):
    self.__decode = MethodType(func, self, StackedAutoEncoder)
"""
##############################################################################
"""
# sda = StackedAutoEncoder(model, gpu=0)
self.sublayer = []

self.model = model
""""""
self.gpu = gpu
if self.gpu >= 0:
    # if using pyCUDA version (v1.2.0 earlier)
    if chainer.__version__ <= '1.2.0':
        chainer.cuda.init(self.gpu)
    # CuPy (1.3.0 later) version
    else:
        chainer.cuda.get_device(self.gpu).use()
    self.model = self.model.to_gpu()
""""""

self.optimizer = optimizers.Adam()
self.optimizer.setup(self.model)
self.loss_function = F.mean_squared_error
self.loss_param = {}
"""
##############################################################################
"""
# sda.set_order(('enc1', 'enc2'), ('dec2', 'dec1'))
def set_order(self, encl, decl):
    if len(encl) != len(decl):
        raise TypeError('Encode/Decode layers mismatch')

    self.depth = len(encl)

    for (el, dl) in zip(encl, reversed(decl)):
        submodel = chainer.FunctionSet(
            enc=self.model[el],
            dec=self.model[dl]
        )
        self.sublayer.append(submodel)
"""
##############################################################################
"""
# sda.set_optimizer(optimizers.AdaDelta)
""""""
def set_optimizer(self, func, param={}):
    self.optimizer = func(**param)
    self.optimizer.setup(self.model)
""""""
self.optimizer = optimizers.AdaDelta()
self.optimizer.setup(self.model)
"""
##############################################################################
"""
# sda.set_encode(encode)
""""""
意味がわからない
encodeを実行してるだけ？
でも引数がない
""""""
def set_encode(self, func):
    self.__encode = MethodType(func, self, StackedAutoEncoder)
def __encode(self, x, layer, train):
    pass

""""""
通常のforward()関数に当たる？
self = model
x = x_data
layer = depth
train = True
""""""
def encode(self, x, layer, train):
    if train:
        x = F.dropout(x, ratio=0.2, train=train)
    if layer == 0:
        return x
    x = F.sigmoid(self.model.enc1(x))
    if layer == 1:
        return x
    x = F.sigmoid(self.model.enc2(x))
    if layer == 2:
        return x
    return x
"""
##############################################################################
"""
# sda.set_decode(decode)
""""""
意味がわからない
decodeを実行してるだけ？
でも引数がない
""""""
def set_decode(self, func):
    self.__decode = MethodType(func, self, StackedAutoEncoder)
def __decode(self, x, layer, train):
    pass

""""""
通常のbackward()関数に当たる？
self = model
x = x_data
""""""
def decode(self, x, layer=None, train=False):
    if not train or layer == 2:
        x = F.sigmoid(self.model.dec2(x))
    if not train or layer == 1:
        x = F.sigmoid(self.model.dec1(x))
    return x
"""
##############################################################################

# err = sda.train(train_data, batchsize=200)

    #err = self.validate(x_batch, layer=l, train=True)
    

##############################################################################



##############################################################################



##############################################################################


"""
encodeメソッド と decodeメソッド
が何やってるか分からない
loss_function なら分かるけど，loss_function() が分からない
set_loss_function() ならまだ分かる
　→ 変数，メソッドの区分けではなく全てオブジェクト

.dot が『word』でグラフ出力出来るはずだけど，
グラフが表示出来ない
　→ Graphvizで表示出来る
"""

"""
forward()
backward()
optimizers
optimizers.setup()
"""


##############################################################################

#print('model.collect_parameters() :', model.collect_parameters())
#print('model.copy_parameters_from() :', model.copy_parameters_from())
#print('model.gradients() :', model.gradients())
#print('model.parameters() :', model.parameters())

#self.sublayer = []
#self.model = model

#self.loss_function = F.mean_squared_error()
#self.loss_param = {}

sublayer = []

loss_function = F.mean_squared_error
loss_param = {}

##############################################################################

#def set_order(self, encl, decl):
#    if len(encl) != len(decl):
#        raise TypeError('Encode/Decode layers mismatch')
#
#    #self.depth = len(encl)
#
#    for (el, dl) in zip(encl, reversed(decl)):
#        submodel = chainer.FunctionSet(
#            enc=self.model[el],
#            dec=self.model[dl]
#        )
#        self.sublayer.append(submodel)


#set_order(('enc1', 'enc2'), ('dec2', 'dec1'))


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

#self.optimizer = optimizers.AdaDelta()
#self.optimizer.setup(self.model)

optimizer = optimizers.AdaDelta()
optimizer.setup(model)

##############################################################################

#def __encode(self, x, layer, train):
#    if train:
#        x = F.dropout(x, ratio=0.2, train=train)
#    if layer == 0:
#        return x
#    x = F.sigmoid(self.model.enc1(x))
#    if layer == 1:
#        return x
#    x = F.sigmoid(self.model.enc2(x))
#    if layer == 2:
#        return x
#    return x


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

#def __decode(self, x, layer=None, train=False):
#    if not train or layer == 2:
#        x = F.sigmoid(self.model.dec2(x))
#    if not train or layer == 1:
#        x = F.sigmoid(self.model.dec1(x))
#    return x


def __decode(x, layer=None, train=False):
    if not train or layer == 2:
        x = F.sigmoid(model.dec2(x))
    if not train or layer == 1:
        x = F.sigmoid(model.dec1(x))
    return x

##############################################################################

#def encode(self, x_data, layer=None, train=False):
#    if args.gpu >= 0:
#        x_data = chainer.cuda.to_gpu(x_data)
#
#    x = chainer.Variable(x_data)
#
#    return self.__encode(x, layer, train)


def encode(x_data, layer=None, train=False):
    if args.gpu >= 0:
        x_data = chainer.cuda.to_gpu(x_data)

    x = chainer.Variable(x_data)

    return __encode(x, layer, train)

##############################################################################
"""
def validate(self, x_data, layer=None, train=False):
    targ = self.encode(x_data, layer - 1, train=False)
    code = self.encode(x_data, layer, train=train)
    
    y = self.__decode(code, layer, train=train)
    
    return self.loss_function(targ, y, **self.loss_param)


def train(self, x_data, batchsize=100, action=(lambda: None)):
        errs = []
        N = len(x_data)
        perm = np.random.permutation(N)

        for l in range(1, self.depth + 1):
            self.optimizer.setup(self.sublayer[l - 1])

            sum_error = 0.

            for i in range(0, N, batchsize):
                x_batch = x_data[perm[i:i + batchsize]]

                self.optimizer.zero_grads()
                err = self.validate(x_batch, layer=l, train=True)

                err.backward()
                self.optimizer.update()

                sum_error += float(chainer.cuda.to_cpu(err.data)) * len(x_batch)
                #sum_error += float(err.data) * len(x_batch)
                action()

            errs.append(sum_error / N)

        return tuple(errs)
"""

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
"""
def forward(self, x_data, train=False):
    code = self.encode(x_data, train=train)
    y = self.__decode(code, train=train)
    
    return y


def test(self, x_data, batchsize=100, action=(lambda: None)):
        N = len(x_data)
        perm = np.random.permutation(N)

        sum_error = 0.

        for i in range(0, N, batchsize):
            x_batch = x_data[perm[i:i + batchsize]]
            y = self.forward(x_batch, train=False)

            if args.gpu >= 0:
                x_batch = chainer.cuda.to_gpu(x_batch)
            x = chainer.Variable(x_batch)

            err = self.loss_function(x, y, **self.loss_param)

            sum_error += float(chainer.cuda.to_cpu(err.data)) * len(x_batch)
            action()

        return sum_error / N
"""

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
"""
# save trained network parameters to file
def save_param(self, dst='./network.param.npy'):
    # model.to_cpu() seems to change itself
    # This causes step-by-step saving each epochs with gpu
    param = numpy.array(self.model.to_cpu().parameters)
    numpy.save(dst, param)
    if self.gpu >= 0:
        self.model.to_gpu()

# load pre-trained network parameters from file
def load_param(self, src='./network.param.npy'):
    if not os.path.exists(src):
        raise IOError('specified parameter file does not exists')

    param = numpy.load(src)
    self.model.copy_parameters_from(param)

    # by this process, model parameters to be cpu_array
    if self.gpu >= 0:
        self.model = self.model.to_gpu()
"""

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
"""
enc1_W = [] #
enc2_W = [] #
dec2_W = [] #
dec1_W = [] #

start_time = time.clock()
# Learning loop
for epoch in six.moves.range(1, n_epoch+1):
    print('epoch : %d' % (epoch))

    #err = self.train(train_data, batchsize=200)
    err = train(train_data, batchsize=200)
    print(err)

    perm = np.random.permutation(len(test_data))
    #terr = self.test(test_data[perm][:100])
    terr = test(test_data[perm][:100])
    print(terr)

    enc1_W.append(model.enc1.W) #
    enc2_W.append(model.enc2.W) #
    dec2_W.append(model.dec2.W) #
    dec1_W.append(model.dec1.W) #

    with open('sda.log', mode='a') as f:
        f.write("%d %f %f %f\n" % (epoch, err[0], err[1], terr))

    #self.save_param('sda.param.npy')
    save_param('sda.param.npy')

end_time = time.clock()
print(end_time - start_time)

model.to_cpu()
pickle.dump(model, open('test_SAE.pkl', 'wb'), -1)
"""
##############################################################################

"""
depth = 2
#def set_model

enc = []
dec = []
for i in range(depth):
    enc.append(F.Liner(n_units[i], n_units[i+1]))
    dec.append(F.Liner(n_units[i+1], n_units[i]))
enc = tuple(enc)
dec = tuple(dec)
"""

#rrr = 5
#'test{}'.format(rrr) = 10
#print(test5)

##############################################################################

# draw digit images
def draw_layer(data, index, length):
    column = math.sqrt(length)
    row = math.sqrt(length)
    #column = 15
    #row = math.ceil(length/column)
    size = math.sqrt(data.shape[0])
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

##############################################################################

def image_save(data, str):
    #save_home = './pic_test4/{}.png'.format(str)
    save_home = '{}{}.png'.format(fig_home, str)
    print('plot start : {}.png'.format(str))
    
    plt.style.use('fivethirtyeight')
    size = math.sqrt(data.shape[0])
    plt.figure(figsize=(size, size*1.045))
    
    print('data.shape :', data.shape)
    print('data.shape[0] :', data.shape[0])

    for i in six.moves.range(data.shape[0]):
        draw_layer(data[i], i, data.shape[0])
        
    plt.savefig(save_home, bbox_inches='tight', pad_inches=0.0)
    plt.close() #
    print('plot end : {}.png'.format(str))
    print()

"""
# enc1_Wを可視化
ew1 = np.array(enc1_W[len(enc1_W)-1])
image_save(ew1, 'ew1_images')

# enc2_Wを可視化
ew2 = np.array(enc2_W[len(enc2_W)-1])
image_save(ew2, 'ew2_images')

# dec2_W.Tを可視化
dw2_T = np.array(dec2_W[len(dec2_W)-1]).T
image_save(dw2_T, 'dw2_T_images')

# dec1_W.Tを可視化
dw1_T = np.array(dec1_W[len(dec1_W)-1]).T
image_save(dw1_T, 'dw1_T_images')
"""
##############################################################################
"""
# Neural net architecture
# 途中からデータを流すニューラルネットの構造
def midstream_forward(x_data, layer):
    x = chainer.Variable(x_data.astype(np.float32))
    if layer >= 2:
        x = F.sigmoid(model.dec2(x))
    y = model.dec1(x)
    return y


for i in range(1, depth+1):
    # identity: 単位行列の生成(正方行列), eye: identityと似ているが列数指定出来る
    hidden_x = np.identity(n_units[i])
    hidden_y = midstream_forward(hidden_x, i)
    
    # hidden layerを可視化
    hidden = np.array(hidden_y.data)
    image_save(hidden, 'hL{}_images'.format(i))
"""
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

"""
for i in range(1, depth+1):
    # identity: 単位行列の生成(正方行列), eye: identityと似ているが列数指定出来る
    dec_hidden_x = np.identity(n_units[i])
    dec_hidden_y = dec_forward(hidden_x, i)

    # hidden layerを可視化
    dec_hidden = np.array(dec_hidden_y.data)
    image_save(dec_hidden, 'dec_hL{}_images'.format(i))

    # identity: 単位行列の生成(正方行列), eye: identityと似ているが列数指定出来る
    enc_hidden_x = np.identity(n_units[0])
    enc_hidden_y = enc_forward(enc_hidden_x, i)
    
    # hidden layerを可視化
    enc_hidden_T = np.array(enc_hidden_y.data).T
    image_save(enc_hidden_T, 'enc_hL{}_T_images'.format(i))
"""
##############################################################################

#period = 1000
period = 10
for epoch in six.moves.range(1, int(n_epoch/period)+1):
    #print('epoch : %d' % (epoch))

    for i in six.moves.range(period):
        print('epoch : %d' %((epoch-1)*period +(i+1)))

        #err = self.train(train_data, batchsize=200)
        err = train(train_data, batchsize=200)
        print(err)

        perm = np.random.permutation(len(test_data))
        #terr = self.test(test_data[perm][:100])
        terr = test(test_data[perm][:100])
        print(terr)

        with open('{}sda.log'.format(fig_home), mode='a') as f:
            f.write("%d %f %f %f\n" % ((epoch-1)*period +(i+1), err[0], err[1], terr))

        #self.save_param('sda.param.npy')
        save_param('{}sda.param.npy'.format(fig_home))

    # 可視化メソッド
    ew1 = np.array(model.enc1.W)
    image_save(ew1, 'ew1({})_images'.format(epoch * period))
    ew2 = np.array(model.enc2.W)
    image_save(ew2, 'ew2({})_images'.format(epoch * period))
    dw2_T = np.array(model.dec2.W).T
    image_save(dw2_T, 'dw2_T({})_images'.format(epoch * period))
    dw1_T = np.array(model.dec1.W).T
    image_save(dw1_T, 'dw1_T({})_images'.format(epoch * period))
    #......
    for i in six.moves.range(1, depth+1):
        # identity: 単位行列の生成(正方行列), eye: identityと似ているが列数指定出来る
        dec_hidden_x = np.identity(n_units[i])
        dec_hidden_y = dec_forward(dec_hidden_x, i)
        
        # hidden layerを可視化
        dec_hidden = np.array(dec_hidden_y.data)
        image_save(dec_hidden, 'dec_hL{}({})_images'.format(i, epoch * period))
        
        # identity: 単位行列の生成(正方行列), eye: identityと似ているが列数指定出来る
        enc_hidden_x = np.identity(n_units[0])
        enc_hidden_y = enc_forward(enc_hidden_x, i)
        
        # hidden layerを可視化
        enc_hidden_T = np.array(enc_hidden_y.data).T
        image_save(enc_hidden_T, 'enc_hL{}({})_T_images'.format(i, epoch * period))

    # 消滅する勾配問題検証
    test_ew1 = chainer.Variable(ew1)
    test_dw1_T = chainer.Variable(dw1_T)
    print('Layer1 :', F.mean_squared_error(test_ew1, test_dw1_T).data)
    test_ew2 = chainer.Variable(ew2)
    test_dw2_T = chainer.Variable(dw2_T)
    print('Layer2 :', F.mean_squared_error(test_ew2, test_dw2_T).data)


model.to_cpu()
pickle.dump(model, open('{}test_SAE.pkl'.format(fig_home), 'wb'), -1)

##############################################################################
