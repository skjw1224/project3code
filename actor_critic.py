
import numpy_rbf as rbf

import numpy as np
import scipy as sp
import os
my_CSTR = os.getcwd()

from replay_buffer import ReplayBuffer

SIGMA = 0.05

INITIAL_POLICY_INDEX = 1
ACTOR_BASIS_NUMBERS = 50
CRITIC_BASIS_NUMBERS = 50
BASIS_FCNS = rbf.gaussian

LEARNING_RATE = 1.
GAMMA = 0.99

class ActorCritic(object):
    def __init__(self, env, device):
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.env_epi_length = env.nT

        self.device = device

        self.replay_buffer = ReplayBuffer(env, device, buffer_size=self.env_epi_length, batch_size=self.env_epi_length)
        self.initial_ctrl = InitialControl(env, device)

        self.actor_rbfnet = rbf.RBF(self.s_dim, ACTOR_BASIS_NUMBERS, BASIS_FCNS) # (T, S) --> (T, F)
        self.actor_f_dim = ACTOR_BASIS_NUMBERS

        self.critic_rbfnet = rbf.RBF(self.s_dim, ACTOR_BASIS_NUMBERS, BASIS_FCNS) # (T, S) --> (T, F)
        self.critic_f_dim = CRITIC_BASIS_NUMBERS

        self.actor_mu = np.zeros([self.actor_f_dim, self.a_dim]) # (F, A)
        self.actor_sigma = np.zeros([self.actor_f_dim, self.a_dim]) # (F, A)
        self.critic_theta = np.zeros([self.critic_f_dim, 1]) # (F, 1)


    def ctrl(self, epi, step, x, u):
        if epi < INITIAL_POLICY_INDEX:
            a_val = self.initial_ctrl.controller(step, x, u)
        else:
            a_val = self.choose_action(epi, step, x)

        a_val = np.clip(a_val, -2, 2)
        return a_val

    def choose_action(self, epi, step, x):

        if step == 0:
            actor_phi = self.actor_rbfnet.eval_basis(x)  # (1, F)
            actor_mean = self.compute_actor_mean(actor_phi)
            actor_var = self.compute_actor_var(actor_phi)

            self.action_traj = np.random.multivariate_normal(actor_mean[0], actor_var, [self.env_epi_length]).reshape([-1, self.a_dim]) # (T, A)

        action = self.action_traj[step, :].reshape(1, -1)  # (1, A)
        return action

    def add_experience(self, epi, *single_expr):
        x, u, r, x2, term = single_expr
        self.replay_buffer.add(*[x, u, r, x2, term])

        if term: # In on-policy method, clear buffer when episode ends
            self.train(epi)
            self.replay_buffer.clear()

    def learning_rate_schedule(self, epi):
        self.alpha_amu = LEARNING_RATE / (1 + epi ** 0.5)
        self.alpha_asig = LEARNING_RATE / (1 + epi ** 0.5)
        self.alpha_c = LEARNING_RATE / (1 + epi ** 0.5)

    def compute_actor_mean(self, actor_phi):
        actor_mean = actor_phi @ self.actor_mu  # (1, F) @ (F, A)
        return actor_mean
        # return np.clip(actor_mean, -1, 1)

    def compute_actor_var(self, actor_phi):
        actor_var = SIGMA * np.diag((np.exp(actor_phi @ self.actor_sigma) ** 2 + 1E-4)[0])  # (1, F) @ (F, A) --> (A, A)
        return actor_var
        # return np.clip(actor_var, -1, 1)


    def train(self, epi):
        self.learning_rate_schedule(epi)
        s_traj, a_traj, r_traj, s2_traj, term_traj = self.replay_buffer.sample_sequence()  # T-number sequence
        traj_data = list(zip(s_traj, a_traj, r_traj, s2_traj))

        del_actor_mu_sum = 0.
        del_actor_sigma_sum = 0.
        del_critic_weight_sum = 0.
        epi_cost = 0.

        for single_data in reversed(traj_data):
            del_critic_weight, td, mc, epi_cost = self.compute_critic_grad(single_data, epi_cost)
            del_actor_mu, del_actor_sigma = self.compute_actor_grad(single_data)

            del_actor_mu_sum += del_actor_mu
            del_actor_sigma_sum += del_actor_sigma
            del_critic_weight_sum += del_critic_weight

            del_actor_weight_sum = np.concatenate([del_actor_mu_sum, del_actor_sigma_sum], axis=0)

            # Critic update
            self.critic_theta -= self.alpha_c * del_critic_weight_sum

        # Actor update - Natural policy gradient
        # fisher = del_actor_weight_sum @ del_actor_weight_sum.T
        # try:
        #     fisher_chol = sp.linalg.cholesky(fisher + 1E-4 * np.eye(2 * self.actor_f_dim))
        #     del_actor_weight = sp.linalg.solve_triangular(fisher_chol, sp.linalg.solve_triangular(fisher_chol.T, del_actor_weight_sum, lower=True))  # [2F, A]
        # except np.linalg.LinAlgError:
        #     del_actor_weight = np.linalg.inv(fisher + 1E-2 * np.eye(2 * self.actor_f_dim)) @ del_actor_weight_sum
        #
        #
        # self.actor_mu -= self.alpha_amu * del_actor_weight[:self.actor_f_dim] * td
        # self.actor_sigma -= self.alpha_asig * del_actor_weight[self.actor_f_dim:] * td

            # Actor update - Advantage actor critic, inf hor
            self.actor_mu -= self.alpha_amu * del_actor_mu * td
            self.actor_sigma -= self.alpha_asig * del_actor_sigma * td
            #
        # # Actor update - REINFORCE
        # self.actor_mu -= self.alpha_amu * del_actor_mu_sum * mc
        # self.actor_sigma -= self.alpha_asig * del_actor_sigma_sum * mc

        self.actor_mu = np.clip(self.actor_mu, -10, 10)
        self.actor_sigma = np.clip(self.actor_sigma, -10, 10)
        self.critic_theta = np.clip(self.critic_theta, -10, 10)

        print(np.linalg.norm(self.actor_mu), np.linalg.norm(self.actor_sigma), np.linalg.norm(self.critic_theta))


    def compute_critic_grad(self, single_data, epi_cost):
        x, u, r, x2 = [_.reshape([1, -1]) for _ in single_data]

        critic_phi = self.critic_rbfnet.eval_basis(x)  # (1, F)
        critic_phi_next = self.critic_rbfnet.eval_basis(x2)  # (1, F)

        V_curr = np.clip(critic_phi @ self.critic_theta, 0., 5.)
        V_next = np.clip(critic_phi_next @ self.critic_theta, 0., 5.)

        td = r + GAMMA * V_next - V_curr # (1, 1)

        del_critic_weight = (- critic_phi).T @ td  # (F, 1)

        epi_cost = GAMMA * epi_cost + r
        mc = epi_cost - V_curr
        return del_critic_weight, td, mc, epi_cost

    def compute_actor_grad(self, single_data):
        x, u, r, x2 = [_.reshape([1, -1]) for _ in single_data]

        actor_phi = self.actor_rbfnet.eval_basis(x) # (1, F)
        eps = u - self.compute_actor_mean(actor_phi) # (1, F) @ (F, A)
        actor_var_inv = np.linalg.inv(self.compute_actor_var(actor_phi))  # (A, A)

        dlogpi_dmu = actor_phi.T @ eps @ actor_var_inv # (F, 1) @ (1, A) @ (A, A)
        dlogpi_dsigma = SIGMA * np.repeat(actor_phi, self.a_dim, axis=0).T @ (eps.T @ eps @ actor_var_inv - np.eye(self.a_dim)) # (F, A) @ (A, A)

        return dlogpi_dmu, dlogpi_dsigma

from pid import PID
class InitialControl(object):
    def __init__(self, env, device):
        self.pid = PID(env, device)

    def controller(self, step, x, u):
        return self.pid.pid_ctrl(step, x)