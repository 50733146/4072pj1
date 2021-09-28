import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
pong1r, pong1s = ".\pong1\pong1_rlist.txt", ".\pong1\pong1_Srlist.txt"

pong1_1r, pong1_1s = ".\pong1.1\pong1.1_rlist.txt", ".\pong1.1\pong1.1_Srlist.txt"
pong1_2r, pong1_2s = ".\pong1.2\pong1.2_rlist.txt", ".\pong1.2\pong1.2_Srlist.txt"
pong2r, pong2s = ".\pong2\pong2_rlist.txt", ".\pong2\pong2_Srlist.txt"

p1rd, p1sd = np.loadtxt(pong1r), np.loadtxt(pong1s)
p2rd, p2sd = np.loadtxt(pong2r), np.loadtxt(pong2s)
fig, ax = plt.subplots()
#plt.plot(reward1_data)
ax.plot(p1rd, "r,", label = "pong1")
ax.plot(p1sd, "r", label = "pong1 MSE")
ax.plot(p2rd, "g,", label = "pong2")
ax.plot(p2sd, "g", label = "pong2 MSE")

ax.set(xlabel='episode time', ylabel='episode reward',
       title='Pong1 ' + 'Reward trend graph')
ax.legend()
plt.grid()
fig.savefig('pong2.png', dpi=100)
#plt.savefig('reward.pdf')
plt.show()