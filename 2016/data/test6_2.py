##############################################################################

import matplotlib.pyplot as plt

##############################################################################

workspace = '/Users/yuki/survey/workspace/2016/data/pic/'
fig_home = '{}outline/'.format(workspace)

data_logs = ['{}08/sda.log'.format(workspace),
             '{}16/sda.log'.format(workspace)]

##############################################################################

plt.style.use('ggplot')
plt.figure(figsize=(8,6))
plt.title('Loss of digit recognition.')
plt.ylabel('mean loss')
plt.xlabel('epoch')

for log in data_logs:
    f = open(log, mode='r')
    lines = f.readlines()
    f.close()
    test_loss = [list(map(float, line.split()))[-1] for line in lines]

    plt.plot(range(len(test_loss)), test_loss)
    plt.plot()

plt.legend(['64_loss','196_loss'], loc='upper right')
plt.savefig('{}2.4_loss.png'.format(fig_home), bbox_inches='tight', pad_inches=0.0)
plt.close()

##############################################################################
