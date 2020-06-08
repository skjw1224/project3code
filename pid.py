import numpy as np

class PID(object):
    def __init__(self, env, device):
        self.env = env
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim

    def pid_ctrl(self, step, x):
        if step == 0:
            self.ei = np.zeros([1, self.s_dim])
        ref =  self.env.ref_traj()
        Kp = 2 * np.ones([self.a_dim, self.s_dim])
        Ki = 0.1 * np.ones([self.a_dim, self.s_dim])
        u = Kp @ (x - ref).T + Ki @ self.ei.T
        u = u.T

        self.ei = self.ei + (x - ref)
        return u