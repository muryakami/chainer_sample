##############################################################################

import matplotlib.pyplot as plt

##############################################################################

workspace = '/Users/yuki/survey/workspace/2016/data/pic/'
fig_home = '{}outline/'.format(workspace)

data_logs = ['{}04/sda.log'.format(workspace),
             '{}08/sda.log'.format(workspace),
             '{}12/sda.log'.format(workspace),
             '{}16/sda.log'.format(workspace),
             '{}20/sda.log'.format(workspace),
             '{}24/sda.log'.format(workspace)]

##############################################################################

plt.style.use('ggplot')

for i,log in enumerate(data_logs):
    f = open(log, mode='r')
    lines = f.readlines()
    f.close()
    test_loss = [list(map(float, line.split()))[-1] for line in lines]

    plt.figure(figsize=(8,6))
    plt.title('Loss of digit recognition.')
    plt.ylabel('mean loss')
    plt.xlabel('epoch')

    plt.plot(range(len(test_loss)), test_loss)
    plt.plot()

    plt.legend(['{}_loss'.format(i+1)], loc='upper right')
    plt.savefig('{}{}_loss.png'.format(fig_home, i+1), bbox_inches='tight', pad_inches=0.0)

plt.close()

##############################################################################
