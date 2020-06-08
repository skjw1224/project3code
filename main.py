
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
my_CSTR = os.getcwd()
sys.path.append(my_CSTR)
from env import CstrEnv
from actor_critic import ActorCritic

MAX_EPISODE = 1000

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
device = 'cpu'

env = CstrEnv()
ACcontroller = ActorCritic(env, device)

s_dim = env.s_dim
a_dim = env.a_dim
o_dim = env.o_dim

plt_num = 0

for epi in range(MAX_EPISODE):
    x, y, u, data_type = env.reset()
    u_idx = None
    trajectory = np.zeros([1, s_dim + o_dim + a_dim + 2])  # s + o + a + r + ref

    for i in range(env.nT):

        u = ACcontroller.ctrl(epi, i, x, u)
        x2, y2, u, r, is_term = env.step(x, u)
        ACcontroller.add_experience(epi, x, u, r, x2, is_term)

        xu = np.concatenate((x[0], u[0]))
        xuy = np.concatenate((xu, np.reshape(y2, [-1, ])))
        ref = env.scale(env.ref_traj(), env.ymin, env.ymax)[0]
        xuyr = np.concatenate((xuy, r[0]))
        xuyrr = np.concatenate((xuyr, ref))
        xuyrr = np.reshape(xuyrr, [1, -1])
        trajectory = np.concatenate((trajectory, xuyrr))

        x, y = x2, y2

    trajectory_n = trajectory

    if epi%50 == 0:
        np.savetxt(my_CSTR + '/data/trajectory' + str(epi) + '.txt', trajectory_n, newline='\n')
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        fig = plt.figure(figsize=[20, 12])
        fig.subplots_adjust(hspace=.4, wspace=.5)
        label = [r'$C_{A}$', r'$C_{B}$', r'$T$', r'$T_{Q}$', r'$\frac{v}{V_{R}}$', r'$Q$',
                 r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$', r'$C_{B}$', r'$cost$']
        for j in range(len(label)):
            if label[j] in (r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$'):
                ax = fig.add_subplot(2, 6, j+5)
            else:
                ax = fig.add_subplot(2, 6, j+1)
            ax.plot(trajectory_n[1:, 0], trajectory_n[1:, j+1])
            if j in (1, 8):
                ax.plot(trajectory_n[1:, 0], trajectory_n[1:, -1], ':g')
            plt.ylabel(label[j], size=8)
        plt.savefig(my_CSTR + '/data/episode' + str(epi) + '.png')
        # if epi%5 == 0: plt.show()
        plt_num += 1
    print(epi)
