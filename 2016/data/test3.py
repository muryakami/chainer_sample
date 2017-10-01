import statistics
import math

workspace = '/Users/yuki/survey/workspace/2016/data/'
data_logs = ['{}pic/00/sda.log'.format(workspace),
             '{}pic/04/sda.log'.format(workspace),
             '{}pic/08/sda.log'.format(workspace),
             '{}pic/12/sda.log'.format(workspace),
             '{}pic/16/sda.log'.format(workspace),
             '{}pic/20/sda.log'.format(workspace),
             '{}pic/24/sda.log'.format(workspace)]

for log in data_logs:
    f = open(log)
    lines = f.readlines()
    f.close()

    test_err = [list(map(float, line.split()))[-1] for line in lines]

    print(log)

    sum = 0
    for i in test_err[100:]:
        sum += i
    print(sum/400)

    print(statistics.mean(test_err[:100]))

    temp = test_err[100:]
    print(len(temp))
    sum = 0
    for i in temp:
        sum += i
    mean = sum/400
    print(mean)
    print(statistics.mean(temp))
    #print(temp[199])
    #print(temp[200])
    #print(temp[201])
    #print((temp[199]+temp[200])/2)
    #print((temp[200]+temp[201])/2)
    print(statistics.median(temp))
    #print(statistics.mode(temp))
    sum = 0
    for i in temp:
        sum += (i-mean)**2
        #sum += (i-statistics.mean(temp))**2
    pvariance = sum/400
    print(pvariance)
    print(statistics.pvariance(temp))
    variance = sum/399
    print(variance)
    print(statistics.variance(temp))
    pstdev = math.sqrt(pvariance)
    print(pstdev)
    print(statistics.pstdev(temp))
    stdev = math.sqrt(variance)
    print(stdev)
    print(statistics.stdev(temp))

    print('平均\t\t中央値\t\t分散\t\t標準偏差\t下限\t\t上限\t\t評価幅')
    print('{:10.9f}\t{:10.9f}\t{:10.9f}\t{:10.9f}\t{:10.9f}\t{:10.9f}\t{:10.9f}'.format(
            statistics.mean(temp),
            statistics.median(temp),
            statistics.pvariance(temp),
            statistics.pstdev(temp),
            statistics.mean(temp) - statistics.pstdev(temp)*2,
            statistics.mean(temp) + statistics.pstdev(temp)*2,
            statistics.pstdev(temp)*4)
          )

    print('{:.9f}\t{:.9f}\t{:.9f}\t{:.9f}\t{:.9f}\t{:.9f}\t{:.9f}'.format(
            round(statistics.mean(temp), 10),
            round(statistics.median(temp), 10),
            round(statistics.pvariance(temp), 10),
            round(statistics.pstdev(temp), 10),
            round(statistics.mean(temp) - statistics.pstdev(temp)*2, 10),
            round(statistics.mean(temp) + statistics.pstdev(temp)*2, 10),
            round(statistics.pstdev(temp)*4, 10))
          )


    print()
