import statistics
import math
import numpy as np

workspace = '/Users/yuki/survey/workspace/2016/data/pic2/'
data_logs = ['{}04/sda.log'.format(workspace),
             '{}20/sda.log'.format(workspace),
             '{}26/sda.log'.format(workspace)]

print('最大値\t\t最小値\t\t平均値\t\t中央値\t\t標準偏差\t誤差棒下限\t誤差棒上限')
for log in data_logs:
    f = open(log)
    lines = f.readlines()
    f.close()

    test_err = [list(map(float, line.split()))[-1] for line in lines]
    temp = test_err[100:]

#    max = max(temp)
#    min = min(temp)
    max = np.max(temp)
    min = np.min(temp)
    mean = statistics.mean(temp)
    median = statistics.median(temp)
    pstdev = statistics.pstdev(temp)

    print('{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(
            round(max, 7),
            round(min, 7),
            round(mean, 7),
            round(median, 7),
            round(pstdev, 7),
            round(mean-pstdev, 7),
            round(mean+pstdev, 7))
          )
