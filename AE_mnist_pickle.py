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
fig_home = '{}pic_test/'.format(workspace)

##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
#parser.add_argument('model', nargs='?') ###
parser.add_argument('model') ###
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

# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batchsize = 100

# 学習の繰り返し回数
#n_epoch   = 30
n_epoch   = 10

# 中間層の数
n_units   = 1000

# ノイズ付加有無
noised = False

# MNISTの手書き数字データのダウンロード
# #HOME/scikit_learn_data/mldata/mnist-original.mat にキャッシュされる
print('fetch MNIST dataset')
mnist = fetch_mldata('MNIST original', data_home='~/survey')
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
#temp1 = np.eye(784, 1000)
#temp1 = chainer.Variable(temp1.astype(np.float32))
#temp2 = np.eye(1000, 784).T
#temp1 = chainer.Variable(temp2.astype(np.float32))
#print('二乗誤差テスト :', F.mean_squared_error(temp1, temp2).data)
#print()
print('mnist.data :\n',mnist.data)
print('mnist.data.shape :', mnist.data.shape)
print()
print('y_test :\n',y_test)
print('len(y_test) :', len(y_test))
print('len(y_test[0]) :', len(y_test[0]))
print('y_test.shape :', y_test.shape)
print('y_test.shape[0] :', y_test.shape[0])
print('y_test.shape[1] :', y_test.shape[1])
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
if args.model == '': ###
    model = chainer.FunctionSet(l1=F.Linear(784, n_units),
                                l2=F.Linear(n_units, 784))
else: ###
    model = pickle.load(open(args.model, 'rb')) ###

# GPU使用の時はGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Neural net architecture
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
    #y = model.l1(x)
    #x_hat  = model.l2(y)
    y = F.Linear(model.l1(x))
    x_hat  = F.Linear(model.l2(y))
    # 誤差関数として二乗誤差関数を用いる
    return F.mean_squared_error(x_hat, t)
"""
# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

##############################################################################
"""
l1_W = []
l2_W = []

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
    ###
    print('training N :', N)
    print('training batchsize :', batchsize)
    print('training loop count :', N / batchsize)
    ###
    for i in six.moves.range(0, N, batchsize):
        x_batch = xp.asarray(x_train[perm[i:i+batchsize]])
        y_batch = xp.asarray(y_train[perm[i:i+batchsize]])
        
        optimizer.zero_grads()
        ###
        #print('training x_batch :', x_batch)
        #print('training x_batch.shape :', x_batch.shape)
        #print('training y_batch :', y_batch)
        #print('training y_batch.shape :', y_batch.shape)
        ###
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        train_loss.append(loss.data)
        sum_loss     += float(loss.data) * batchsize

    print('\ttrain mean loss = {}'.format(sum_loss / N))
    
    # evaluation
    sum_loss     = 0
    ###
    print('evaluation N :', N_test)
    print('evaluation batchsize :', batchsize)
    print('evaluation loop count :', N_test / batchsize)
    ###
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = xp.asarray(x_test[i:i+batchsize])
        y_batch = xp.asarray(y_test[i:i+batchsize])
        ###
        #print('evaluation x_batch :', x_batch)
        #print('evaluation x_batch.shape :', x_batch.shape)
        #print('evaluation y_batch :', y_batch)
        #print('evaluation y_batch.shape :', y_batch.shape)
        ###
        # 評価時には dropout しないから, train=False
        loss = forward(x_batch, y_batch, train=False)

        test_loss.append(loss.data)
        sum_loss     += float(loss.data) * batchsize

    loss_val = sum_loss / N_test #batchsize(100)ごとのlossの合計(10000/100回足してる)/testデータの数(10000)
    
    print('\ttest  mean loss = {}'.format(loss_val))
    if epoch == 1:
        loss_std = loss_val
        loss_rate.append(100)
    else:
        print('\tratio : %.3f'%(loss_val/loss_std * 100))
        loss_rate.append(loss_val/loss_std * 100)
        
    if prev_loss >= 0:
        diff = loss_val - prev_loss
        ratio = diff/prev_loss * 100
        print('\timpr rate : %.3f'%(-ratio))
    
    prev_loss = sum_loss / N_test
    test_mean_loss.append(loss_val)
    
    l1_W.append(model.l1.W)
    l2_W.append(model.l2.W)
    end_time = time.clock()
    print('\ttime = %.3f' %(end_time-start_time))

###
#print('l1_W :\n',l1_W)
#print('len(l1_W) :', len(l1_W))
#print('l2_W :\n',l2_W)
#print('len(l2_W) :', len(l2_W))
#print('l3_W :', l3_W)
#print('len(l3_W) :', len(l3_W))
###

# Draw mean loss graph
plt.style.use('ggplot')
plt.figure(figsize=(10,7))
plt.plot(test_mean_loss, lw=1)
plt.title('mean loss graph')
plt.ylabel('mean loss')
plt.xlabel('epoch')
plt.savefig('{}mean loss graph.png'.format(fig_home))
plt.close() #

model.to_cpu()
pickle.dump(model, open('model.pkl', 'wb'), -1)
# pickle.dump(model, open('{}model.pkl'.format(fig_home), 'wb'), -1)
"""
##############################################################################
"""
###
print()
print('start')
print('len(l1_W)) : ',len(l1_W))
print('len(l1_W[len(l1_W)-1]) : ',len(l1_W[len(l1_W)-1]))
print('len(l1_W[len(l1_W)-1][0]) : ',len(l1_W[len(l1_W)-1][0]))
print('len(l1_W[len(l1_W)-1][len(l1_W[len(l1_W)-1])]) : ',len(l1_W[len(l1_W)-1][len(l1_W)-1]))
print()
print('len(l2_W)) : ',len(l2_W))
print('len(l2_W[len(l2_W)-1]) : ',len(l2_W[len(l2_W)-1]))
print('len(l2_W[len(l2_W)-1][0]) : ',len(l2_W[len(l2_W)-1][0]))
print('len(l2_W[len(l2_W)-1][len(l2_W[len(l2_W)-1])]) : ',len(l2_W[len(l2_W)-1][len(l2_W)-1]))
print('end')
print()
###
"""
##############################################################################
"""
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
    h1 = F.relu(model.l1(chainer.Variable(xxx.reshape(1,784))))
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
plt.close() #
print('plot end : IO_images.png')
print()
"""
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
    plt.close() #
    print('plot end : {}.png'.format(str))
    print()

"""
# W(1)を可視化
w1 = np.array(l1_W[len(l1_W)-1])
image_save(w1, 'w1_images')

# W(2).Tを可視化
w2_T = np.array(l2_W[len(l2_W)-1]).T
image_save(w2_T, 'w2_T_images')

# 消滅する勾配問題検証
test_w1 = chainer.Variable(w1)
test_w2_T = chainer.Variable(w2_T)
#test_w1 = chainer.Variable(w1.astype(np.float32))
#test_w2_T = chainer.Variable(w2_T.astype(np.float32))
print('Layer1 :', F.mean_squared_error(test_w1, test_w2_T))
print('Layer1 :', F.mean_squared_error(test_w1, test_w2_T).data)
print('Layer1 :', F.mean_squared_error(test_w1, test_w2_T).data.shape)

##############################################################################

# Neural net architecture
# 途中からデータを流すニューラルネットの構造
# midstream_forward:
def dec_forward(x_data):
    x = chainer.Variable(x_data.astype(np.float32))
    y = model.l2(x)
    return y

# identity: 単位行列の生成(正方行列), eye: identityと似ているが列数指定出来る
dec_hidden_x = np.identity(n_units)
dec_hidden_y = dec_forward(dec_hidden_x)

# hidden layerを可視化
dec_hidden = np.array(dec_hidden_y.data)
image_save(dec_hidden, 'dec_hL_images')

##############################################################################

# Neural net architecture
# 途中からデータを流すニューラルネットの構造
def enc_forward(x_data):
    x = chainer.Variable(x_data.astype(np.float32))
    y = model.l1(x)
    return y

# identity: 単位行列の生成(正方行列), eye: identityと似ているが列数指定出来る
enc_hidden_x = np.identity(784)
enc_hidden_y = enc_forward(enc_hidden_x)

# hidden layerを可視化
enc_hidden_T = np.array(enc_hidden_y.data).T
image_save(enc_hidden_T, 'enc_hL_T_images')

##############################################################################

test_enc_hidden_T = chainer.Variable(enc_hidden_T)
print('一致率')
print('w1_images, enc_hidden_T : ', F.mean_squared_error(test_w1, test_enc_hidden_T).data)
"""
##############################################################################
##############################################################################

def dec_forward(x_data):
    x = chainer.Variable(x_data.astype(np.float32))
    y = model.l2(x)
    return y

def enc_forward(x_data):
    x = chainer.Variable(x_data.astype(np.float32))
    y = model.l1(x)
    return y

train_loss = []
test_loss = []
test_mean_loss = []

prev_loss = -1
loss_std = 0

loss_rate = []

period = 5
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
            
            train_loss.append(loss.data)
            sum_loss     += float(loss.data) * batchsize
            
        print('\ttrain mean loss = {}'.format(sum_loss / N))
        
        # evaluation
        sum_loss     = 0
        for i in six.moves.range(0, N_test, batchsize):
            x_batch = xp.asarray(x_test[i:i+batchsize])
            y_batch = xp.asarray(y_test[i:i+batchsize])
            
            loss = forward(x_batch, y_batch, train=False)
            
            test_loss.append(loss.data)
            sum_loss     += float(loss.data) * batchsize
            
        loss_val = sum_loss / N_test #batchsize(100)ごとのlossの合計(10000/100回足してる)/testデータの数(10000)
        
        print('\ttest  mean loss = {}'.format(loss_val))
        if epoch == 1:
            loss_std = loss_val
            loss_rate.append(100)
        else:
            print('\tratio : %.3f'%(loss_val/loss_std * 100))
            loss_rate.append(loss_val/loss_std * 100)
            
        if prev_loss >= 0:
            diff = loss_val - prev_loss
            ratio = diff/prev_loss * 100
            print('\timpr rate : %.3f'%(-ratio))
        
        prev_loss = sum_loss / N_test
        test_mean_loss.append(loss_val)
        
        end_time = time.clock()
        print('\ttime = %.3f' %(end_time-start_time))

    # 可視化メソッド
    ew1 = np.array(model.l1.W)
    image_save(ew1, 'ew1({})_images'.format(epoch * period))
    dw1_T = np.array(model.l2.W).T
    image_save(dw1_T, 'dw1_T({})_images'.format(epoch * period))
    # ......
    dec_hidden_x = np.identity(n_units)
    dec_hidden_y = dec_forward(dec_hidden_x)
    dec_hidden = np.array(dec_hidden_y.data)
    image_save(dec_hidden, 'dec_hL1({})_images'.format(epoch * period))
    enc_hidden_x = np.identity(784)
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


# Draw mean loss graph
plt.style.use('ggplot')
plt.figure(figsize=(10,7))
plt.plot(test_mean_loss, lw=1)
plt.title('mean loss graph')
plt.ylabel('mean loss')
plt.xlabel('epoch')
plt.savefig('{}mean loss graph.png'.format(fig_home))
plt.close() #

model.to_cpu()
# pickle.dump(model, open('model.pkl', 'wb'), -1)
pickle.dump(model, open('{}model.pkl'.format(fig_home), 'wb'), -1)

##############################################################################

"""
# 平均二乗誤差平方根
def RMSE(x_data, y_data):
    sum = 0
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            sum = sum + (x_data[i][j] - y_data[i][j]) * (x_data[i][j] - y_data[i][j])
    return math.sqrt(sum / (x_data.shape[0] * x_data.shape[1]))

#square_error = [(x-y)**2 for x in x_data for y in y_data]
#sum_square_error = 
#mean_square_error =  for e 
"""
