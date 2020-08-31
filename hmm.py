from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        alpha[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            for i in range(S):
                alpha[i, t] = self.B[i, self.obs_dict[Osequence[t]]] * np.sum(self.A[:, i] * alpha[:, (t-1)])
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        beta[:, L-1] = 1
        for t in range(L-2, -1, -1):
            for i in range(S):
                beta[i, t] = sum(self.A[i, :] * self.B[:, self.obs_dict[Osequence[t+1]]] * beta[:, (t+1)])
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        prob = np.sum(self.forward(Osequence), axis=0)[-1]
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        this_alpha = self.forward(Osequence)
        this_beta = self.backward(Osequence)
        denominator = self.sequence_prob(Osequence)
        prob = np.divide(this_alpha * this_beta, denominator)
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        this_alpha = self.forward(Osequence)
        this_beta = self.backward(Osequence)
        denominator = self.sequence_prob(Osequence)
        for t in range(L-1):
            for i in range(S):
                prob[i, :, t] = np.divide(this_alpha[i, t] * self.A[i, :] * self.B[:, self.obs_dict[Osequence[t+1]]] * this_beta[:, t+1], denominator)
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        S = len(self.pi)
        L = len(Osequence)
        new_state_dict = {v : k for k, v in self.state_dict.items()}
        delta = np.zeros((S, L))
        Delta = np.zeros((S, L), dtype=int)
        delta[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        Delta[:, 0] = 0
        for t in range(1, L):
            for i in range(S):
                all_mediate = np.zeros(S)
                for j in range(S):
                    this_mediate = self.A[j, i] * delta[j, t-1]
                    all_mediate[j] = this_mediate
                delta[i, t] = self.B[i, self.obs_dict[Osequence[t]]] * np.max(all_mediate)
                Delta[i, t] = np.argmax(all_mediate)
        path.append(new_state_dict[np.argmax(delta[:, -1])])
        for t in range(L-1, 0, -1):
            path.append(new_state_dict[Delta[self.state_dict[path[-1]], t]])
        path = path[::-1]
        ###################################################
        return path
