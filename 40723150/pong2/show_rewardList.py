import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
file1_name = "pong2_rlist_10K.txt"
file1_sum_name = "pong2_Srlist_10K.txt"
file2_name = "pong2_r_rlist.txt"
file2_sum_name = "pong2_r_Srlist.txt"
reward1_data = np.loadtxt(file1_name)
reward1_sum_data = np.loadtxt(file1_sum_name)
reward2_data = np.loadtxt(file2_name)
reward2_sum_data = np.loadtxt(file2_sum_name)
fig, ax = plt.subplots()
#plt.plot(reward1_data)
ax.plot(reward1_data, "r,", label = "pong2")
ax.plot(reward1_sum_data, "r", label = "pong2 MSE")
ax.plot(reward2_data, "c,", label = "pong2_r")
ax.plot(reward2_sum_data, "c", label = "pong2_r MSE")
ax.set(xlabel='episode time', ylabel='episode reward',
       title='Pong2 ' + 'Reward trend graph')
ax.legend()
plt.grid()
fig.savefig('pong2.png', dpi=100)
#plt.savefig('reward.pdf')
plt.show()