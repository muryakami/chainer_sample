workspace = '/Users/yuki/survey/workspace/2016/data/'
data_logs = ['{}pic/DAE/sda.log'.format(workspace),
             '{}pic/00/sda.log'.format(workspace),
             '{}pic/04/sda.log'.format(workspace),
             '{}pic/06/sda.log'.format(workspace),
             '{}pic/08/sda.log'.format(workspace),
             '{}pic/12/sda.log'.format(workspace),
             '{}pic/16/sda.log'.format(workspace),
             '{}pic/20/sda.log'.format(workspace),
             '{}pic/24/sda.log'.format(workspace)]

# w = open('write.txt', 'w') #
for log in data_logs:
    f = open(log)
    lines = f.readlines()
    f.close()

    test_err = [list(map(float, line.split()))[-1] for line in lines]

    print(log)
    #print(*test_err)
    #print('length :', len(test_err))
    #print('length-300 :', len(test_err[300:]))

    sum = 0
    for i in test_err[300:]:
        sum += i
    #print('average :', sum/200)
    print(sum/200)
    sum = 0
    for i in test_err[250:]:
        sum += i
    print(sum/250)
    sum = 0
    for i in test_err[100:]:
        sum += i
    print(sum/400)
    print()

    """
    w.write(log)
    w.write('\n')
    w.writelines(str(test_err))
    w.write('\n')
    w.write('length : '+str(len(test_err)))
    w.write('\n')
    w.write('length-300 : '+str(len(test_err[300:])))
    w.write('\n')

    sum = 0
    for i in test_err[300:]:
        sum += i
    w.write('average : '+str(sum/200))
    w.write('\n')

w.close()
"""
