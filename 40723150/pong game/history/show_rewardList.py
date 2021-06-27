import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
file1_name = "pong1.2_reward_list.txt"
file1_sum_name = "pong1.2_sum_reward_list.txt"
file2_name = "reward_list.txt"
file2_sum_name = "sum_reward_list.txt"
reward1_data = np.loadtxt(file1_name)
reward1_sum_data = np.loadtxt(file1_sum_name)
reward2_data = np.loadtxt(file2_name)
reward2_sum_data = np.loadtxt(file2_sum_name)
fig, ax = plt.subplots()
#plt.plot(reward1_data)
ax.plot(reward2_data, "g,", label = "pong2")
ax.plot(reward1_data, "r,", label = "pong1.2")
ax.plot(reward2_sum_data, "b", label = "pong2 sum")
ax.plot(reward1_sum_data, "c", label = "pong1.2 sum")

ax.set(xlabel='episode time', ylabel='episode reward',
       title='pong2 V.S. pong1.2\n' + 'Reward trend graph')
ax.legend()
plt.grid()
#fig.savefig("pong1.1_reward.png", dpi=100)
#plt.savefig('reward.pdf')
plt.show()