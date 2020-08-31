import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	N_sentence = len(train_data)
	S = len(tags)
	pi = np.zeros(S)
	A = np.zeros((S, S))
	B = np.array([[] for _ in range(S)])
	obs_dict = {}
	state_dict = {}
	add = np.array([0 for _ in range(S)])
	for i in range(S):
		state_dict[tags[i]] = i

	for i in range(N_sentence):
		for j in range(train_data[i].length):
			this_word = train_data[i].words[j]
			this_tag = train_data[i].tags[j]
			if this_word not in obs_dict.keys():
				obs_dict[this_word] = len(obs_dict)
				B = np.column_stack((B, add.T))

			if j != 0:
				before_this_tag = train_data[i].tags[j-1]
				A[state_dict[before_this_tag], state_dict[this_tag]] += 1
			else:
				pi[state_dict[this_tag]] += 1

			B[state_dict[this_tag], obs_dict[this_word]] += 1

	pi = pi / np.sum(pi)
	A = (A.T / np.sum(A, axis=1)).T
	B = (B.T / np.sum(B, axis=1)).T
	model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	N_sentence = len(test_data)
	add = np.array([1e-16 for _ in range(len(model.state_dict))])
	for i in range(N_sentence):
		for j in range(test_data[i].length):
			this_word = test_data[i].words[j]
			if this_word not in model.obs_dict.keys():
				model.obs_dict[this_word] = len(model.obs_dict)
				model.B = np.column_stack((model.B, add))
		tagging.append(model.viterbi(test_data[i].words))
	###################################################
	return tagging
