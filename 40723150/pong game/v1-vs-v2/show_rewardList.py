import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
file1_name = "pong1_rlist.txt"
file1_sum_name = "pong1_Srlist.txt"
file2_name = "pong1.2_rlist.txt"
file2_sum_name = "pong1.2_Srlist.txt"
file3_name = "pong2_rlist.txt"
file3_sum_name = "pong2_Srlist.txt"
reward1_data = np.loadtxt(file1_name)
reward1_sum_data = np.loadtxt(file1_sum_name)
reward2_data = np.loadtxt(file2_name)
reward2_sum_data = np.loadtxt(file2_sum_name)
reward3_data = np.loadtxt(file3_name)
reward3_sum_data = np.loadtxt(file3_sum_name)
fig, ax = plt.subplots()
#plt.plot(reward1_data)
ax.plot(reward1_data, "c,", label = "pong1")
ax.plot(reward2_data, "b,", label = "pong1.2")
ax.plot(reward3_data, "r,", label = "pong2")
ax.plot(reward1_sum_data, "c", label = "pong1 sum")
ax.plot(reward2_sum_data, "b", label = "pong1.2 sum")
ax.plot(reward3_sum_data, "r", label = "pong2 sum")

ax.set(xlabel='episode time', ylabel='episode reward',
       title='pong1, pong1.2 and pong2 ' + 'Reward trend graph')
ax.legend()
plt.grid()
fig.savefig("pong1_pong1.2_pong2_reward.png", dpi=100)
#plt.savefig('reward.pdf')
plt.show()