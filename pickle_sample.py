# 例えば、mnistの例の場合、モデルを読み込む部分は
if args.model == '':
    model = chainer.FunctionSet(l1=F.Linear(784, n_units),
                                l2=F.Linear(n_units, n_units),
                                l3=F.Linear(n_units, 10))
else:
    with open(args.model, 'rb') as i:
        model = pickle.load(i)
# となり、モデルを書き込む部分は、スクリプトの最後に
model.to_cpu()
with open('model.pkl', 'wb') as o:
    pickle.dump(model, o)

#################################################################

# Read
if args.model == '':
    model = chainer.FunctionSet(l1=F.Linear(784, n_units),
                                l2=F.Linear(n_units, n_units),
                                l3=F.Linear(n_units, 10))
else:
    model = pickle.load(open(args.model, 'rb'))
# Write
model.to_cpu()
pickle.dump(model, open('model.pkl', 'wb'))
