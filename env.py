import numpy as np
from os import path
from scipy.integrate import solve_ivp
from functools import partial

class CstrEnv(object):
    # Certain paramaters
    E1 = -9758.3
    E2 = -9758.3
    E3 = -8560.
    rho = 0.9342  # (KG / L)
    Cp = 3.01  # (KJ / KG K)
    kw = 4032.  # (KJ / h M ^ 2 K)
    AR = 0.215  # (M ^ 2)
    VR = 10.  # L
    mk = 5.  # (KG)
    CpK = 2.0  # (KJ / KG K)
    CA0 = 5.1  # mol / L
    T0 = 378.05  # K

    # Real values of uncertain parameters
    k10 = 1.287e+12
    k20 = 1.287e+12
    k30 = 9.043e+9
    delHRab = 4.2  # (KJ / MOL)
    delHRbc = -11.0  # (KJ / MOL)
    delHRad = -41.85  # (KJ / MOL)

    k10_mu_prior = 1.327e+12
    k20_mu_prior = 1.247e+12
    k30_mu_prior = 8.773e+9
    delHRab_mu_prior = 1.84
    delHRbc_mu_prior = -9.09
    delHRad_mu_prior = -43.26



    def __init__(self):
        # Dimensions and variables
        self.s_dim = 7
        self.a_dim = 2
        self.o_dim = 1

        self.x0 = np.array([[0., 2.1404, 1.20, 387.34, 386.06, 14.19, -1113.5]])
        self.u0 = np.array([[0., 0.]])
        self.t0 = 0.
        self.dt = 20 / 3600.  # h
        self.tT = 3600 / 3600
        self.nT = int(self.tT / self.dt) + 1


        self.Q = np.diag([10])
        self.R = np.diag([0.01, 0.01])
        self.H = np.diag([10])

        self.xmin = np.array([[self.t0, 0.001, 0.001, 353.15, 363.15, 3, -9000]])
        self.xmax = np.array([[self.tT, 3.5, 1.4, 413.15, 408.15, 35, 0]])
        self.ymin = np.array([[self.xmin[0, 2]]])
        self.ymax = np.array([[self.xmax[0, 2]]])
        self.umin = np.array([[-0.5, -200]]) / self.dt
        self.umax = np.array([[0.5, 200]]) / self.dt

        # partial function: Pre declaration of the Leftmost arguments
        self.dx_eval = self.system_functions
        self.y_eval = self.output_functions
        self.c_eval = partial(self.cost_functions, False)
        self.cT_eval = partial(self.cost_functions, True)

        self.reset()

    def reset(self):
        state = self.scale(self.x0, self.xmin, self.xmax)
        action = self.scale(self.u0, self.umin, self.umax)
        obsv = self.y_eval(state, action)
        data_type = 'path'

        return state, obsv, action, data_type


    def step(self, state, action):

        # Scaled state, action, output
        t = self.descale(state, self.xmin, self.xmax)[0][0]
        x = state
        u = action

        # Identify data_type
        if t <= self.tT - 0.5 * self.dt:  # leg_BC assigned & interior time --> 'path'
            data_type = 'path'
        else:
            data_type = 'terminal'

        # Integrate ODE
        if data_type == 'path':
            # input: x, u: [1, s] --> odeinput: x: [s, ] --> output: x: [1, s]
            dx = lambda t, x: self.dx_eval(t, x, u)
            xvec = np.reshape(x, [-1, ])
            sol_x = solve_ivp(dx, [t, t + self.dt], xvec, method='LSODA')
            xplus = np.reshape(sol_x.y[:, -1], [1, -1])

            xplus = np.clip(xplus, -2, 2)

            costs = self.c_eval(xplus, u) * self.dt

            # Terminal?
            is_term = False # Use consistent dimension [1, 1]

        else: # data_type = 'terminal'
            xplus = x
            costs = self.cT_eval(xplus, u) * self.dt

            is_term = True # Use consistent dimension [1, 1]

        yplus = self.y_eval(xplus, u)

        return xplus, yplus, u, costs, is_term

    def ref_traj(self):
        return np.array([[0.95]])

    def system_functions(self, t, x, u):
        x = self.descale(x, self.xmin, self.xmax)
        u = self.descale(u, self.umin, self.umax)

        E1 = self.E1
        E2 = self.E2
        E3 = self.E3

        rho = self.rho
        Cp = self.Cp
        kw = self.kw
        AR = self.AR
        VR = self.VR
        mk = self.mk
        CpK = self.CpK

        CA0 = self.CA0
        T0 = self.T0


        t, CA, CB, T, TK, VdotVR, QKdot = np.reshape(x, [-1, ])
        dVdotVR, dQKdot = np.reshape(u, [-1, ])
        k10, k20, k30, delHRab, delHRbc, delHRad = self.k10, self.k20, self.k30, self.delHRab, self.delHRbc, self.delHRad

        k1 = k10 * np.exp(E1 / T)
        k2 = k20 * np.exp(E2 / T)
        k3 = k30 * np.exp(E3 / T)

        dx = [1.,
              VdotVR * (CA0 - CA) - k1 * CA - k3 * CA ** 2.,
              -VdotVR * CB + k1 * CA - k2 * CB,
              VdotVR * (T0 - T) - (k1 * CA * delHRab + k2 * CB * delHRbc + k3 * CA ** 2. * delHRad) / (rho * Cp) + (
                          kw * AR) / (rho * Cp * VR) * (TK - T),
              (QKdot + (kw * AR) * (T - TK)) / (mk * CpK),
              dVdotVR,
              dQKdot]

        dx = np.reshape(dx, [1, -1])
        dx = self.scale(dx, self.xmin, self.xmax, shift=False)

        return dx

    def output_functions(self, x, u):
        x = self.descale(x, self.xmin, self.xmax)
        u = self.descale(u, self.umin, self.umax)

        x = np.reshape(x, [-1, ])
        y = x[2]
        y = np.reshape(y, [1, -1])

        y = self.scale(y, self.ymin, self.ymax, shift=True)
        return y

    def cost_functions(self, is_terminal, *args):
        x, u = args
        Q = self.Q
        R = self.R
        H = self.H

        y = self.output_functions(x, u)
        ref = self.scale(self.ref_traj(), self.ymin, self.ymax)

        if not is_terminal:
            cost = (y - ref) @ Q @ (y - ref).T + u @ R @ u.T
        else: # terminal condition
            cost = (y - ref) @ H @ (y - ref).T

        return cost

    def scale(self, var, min, max, shift=True): # [min, max] --> [-1, 1]
        shifting_factor = max + min if shift else 0.
        scaled_var = (2. * var - shifting_factor) / (max - min)
        return scaled_var

    def descale(self, scaled_var, min, max): # [-1, 1] --> [min, max]
        var = (max - min) / 2 * scaled_var + (max + min) / 2
        return var

