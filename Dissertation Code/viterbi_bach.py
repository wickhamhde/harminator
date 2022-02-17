import random
import numpy as np
import harmIO
import helper as hp

def init_matrices(state_space, emission_space):
	#Transisiton Matrix
	transition = np.zeros((len(state_space),len(state_space)))
	#Emission Matrix
	emission = np.zeros((len(state_space), len(emission_space)))
	#Starting probabilities
	initial = [0 for _ in range(len(state_space))]

	return transition, emission, initial


def update_matrices(transition, emission, initial, state_space, emission_space, observed, state_seq):


	#Update starting probability list with probability of first state
	initial[state_space.index(state_seq[0])] += 1


	#Run through every line
	for n in range(1,len(observed)):

		#Current state and observation
		currentObs = observed[n]
		currentState = state_seq[n]
		lastState = state_seq[n-1]

		#Update emission probabilities
		
		#Check if the current time step is not merely a continuation of held notes
		if (hp.checkNew(currentObs) or hp.checkNew(currentState)) == True:
			

			#Find the index position for the current state
			currentStatePos = state_space.index(hp.raw(currentState))

			#Find index position for current emission
			currentObsPos = emission_space.index(hp.raw((currentObs)))

			#Find the index position for the current state
			lastStatePos = state_space.index(hp.raw(lastState))
			#Increment correct matrix entries
			emission[currentStatePos][currentObsPos] += 1
			transition[lastStatePos][currentStatePos] += 1	
	

	return transition, emission, initial


def update_matrices_order2(transition, emission, initial, state_space, emission_space, observed, state_seq):

	#Update starting probability list with probability of first state
	initial[state_space.index(state_seq[0])] += 1		
	
	#Run through every line
	for n in range(1,len(observed)):

		#Current state and observation
		currentObs = observed[n]
		currentState = state_seq[n]
		stateMinus1 = state_seq[n-1]

		#Update emission probabilities
		
		#Find the index position for the current state
		currentStateInd = state_space.index(hp.raw(currentState))
		#Find index position for current emission
		currentObsInd = emission_space.index(hp.raw((currentObs)))
		#Increment emission matrix entry
		emission[currentStateInd][currentObsInd] += 1

		#Find index of last state
		stateMinus1Ind = state_space.index(hp.raw(stateMinus1))
		#Increment transition matrix entry
		transition[stateMinus1Ind][currentStateInd] += 1

		if n > 1:
			stateMinus2 = state_seq[n-2]

			#Do not need to increment transition between stateMinus2 and stateMinus1 since it would have already been done in last iteration of the for loop

			compound = (stateMinus2, stateMinus1)

			#Find index of compound state
			compoundInd = state_space.index(compound)

			#Increment transition matrix entry
			transition[compoundInd][currentStateInd] += 1			


	return transition, emission, initial

def backOffEmission(obs, emission_space, compound=False):
	#Backing off function for compound observations
	if compound == True:
		if obs in emission_space:
			return obs
		else:
			if obs[0] not in emission_space and hp.enharmonic(obs[0]) not in emission_space:
				print('Serious probs', obs)
			return obs[0]
	else:
		# #Case for unseen unusual ornamentation in soprano line, back off to plain crotchet
		if isinstance(obs, tuple):
			if len(obs) == 4:
				if obs not in emission_space:
					newObs = []
					newObs.append(obs[0])
					for i in range(3):
						newObs.append(hp.dashed(newObs[0]))
					return tuple(newObs)
		return obs


def ornament0Prob(obs, state_space, t1, t2, i, j):
	
	state =  (0,0,0,0)

	stateInd = state_space.index(state)

	t1[j][i] = 0.001
	t2[j][i] = stateInd

	return t1, t2


def dynamicAddEmission(item, e_matrix, emission_space, state_space):
	if item not in emission_space:
		#Catch generated emission that has not appeared in training set
		emission_space.append(item)
		new_e_matrix = np.zeros((len(state_space), len(emission_space)))
		for a in range(len(state_space)):
			for b in range(len(emission_space)-1):
				new_e_matrix[a][b] = e_matrix[a][b]
			new_e_matrix[a][-1] = 1
		e_matrix = new_e_matrix

	return e_matrix, emission_space

def viterbi(observed, state_space, emission_space , t_matrix, e_matrix, start_probs, compound=False, orn=False, verbose=False):

	#Number of states
	E = len(e_matrix)
	K = len(start_probs)
	T = len(observed)

	#Table for storing max probability for each possible visited hidden state
	t1 = [[0 for _ in range(T)] for _ in range(K)]
	#Table for storing paramaters for achieving T1's probability values
	t2 = [[0 for _ in range(T)] for _ in range(K+1)]

	#Calculate first states' probabilities from starting state
	for i in range(K):
		obs = backOffEmission(observed[0], emission_space, compound)
		#P(State = state i) = P(Transition from starting state to state i) * P(State i emitting first observation)
		t1[i][0] = start_probs[i]*e_matrix[i][emission_space.index(obs)]

		t2[i][0] = 0
	#Iterate through remaining observations
	for t in range(1, T):
		obs = backOffEmission(observed[t], emission_space, compound)
		if verbose:
			print('State', t, 'observation is', obs, 'taken from', observed[t])

		#Iterate through possible hidden states at time t
		for j in range(K):
			temp1 = []
			temp2 = []
			#Iterate through possible transitions from state at time t-1 to current state
			for k in range(K):

				#t1[k][t-1] = probability of path up to previous hidden state
				#t_matrix[k][j] = the transition probability from previous state (k) to current state(j)
				#e_matrix[j][observed[t]] = the emission probability to the observed state from current state (j)
				prob_last_state = t1[k][t-1]
				prob_transition = t_matrix[k][j]
				try:
					prob_emission = e_matrix[j][emission_space.index(obs)]
				except:
					print(obs, observed, emission_space)



				temp1.append(prob_last_state*prob_transition*prob_emission)
				temp2.append(k)


			if max(temp1) == 0 and orn == True:
				t1, t2 = ornament0Prob(obs, state_space, t1, t2, t, j)
			else:
			# Store best paths to each hidden state so far and their probabilities
				t1[j][t] = max(temp1)
				t2[j][t] = temp1.index(max(temp1))


	ztemp = []
	for k in range(len(t1)):
		ztemp.append(t1[k][T-1])

	Z = [0 for t in range(T)]
	Z[-1] = ztemp.index(max(ztemp))

	X = [0 for t in range(T)]
	X[-1] = state_space[Z[-1]]

	trange = reversed(range(T))

	for t in trange:
		Z[t-1] = t2[Z[t]][t]
		X[t-1] = state_space[Z[t-1]]
	
	return X
	
def sliceProb(stateMinus2, stateMinus1, currentState, observed, t, transition, emission,state_space_singles, state_space, emission_space, comp):
	#Find the probability of a pair of symbols to be treated as the probility for single state, utilising backing off
	#stateMinus2 is state at t-2, stateMinus1 is state at t-1

	compound = (stateMinus2, stateMinus1)
	
	obs = backOffEmission(observed[t], emission_space, comp)

	sliced_prob = 1

	if compound in state_space:
		compInd = state_space.index(compound)
		currentInd = state_space.index(currentState)
		obsInd = emission_space.index(obs)

		if isinstance(currentState, tuple):
			print(currentState)

		multiplier1 = transition[compInd][currentInd]
		multiplier2 = emission[currentInd][obsInd]

		if sliced_prob*multiplier1*multiplier2 != 0:
			return sliced_prob*multiplier1*multiplier2
	#Back off to order 1 if necessary
	stateMinus2Ind = state_space.index(stateMinus2)
	stateMinus1Ind = state_space.index(stateMinus1)
	currentInd = state_space.index(currentState)

	obsInd = emission_space.index(obs)

	# sliced_prob *= transition[stateMinus2Ind][stateMinus1Ind]
	# sliced_prob *= emission[stateMinus1Ind][obsMinus1Ind]
	sliced_prob *= transition[stateMinus1Ind][currentInd]
	sliced_prob *= emission[currentInd][obsInd] 
	
	return sliced_prob


def viterbi_order_2(observed, state_space, emission_space, t_matrix, e_matrix, start_probs, compound):

	#Isolate single elements in state space
	state_space_singles = []
	for item in state_space:
		if isinstance(item, str):
			state_space_singles.append(item)

	#Number of hidden states or combinations of them
	K = len(state_space_singles)
	#No. of time steps
	T = len(observed)

	#Table for storing max probability for each possible visited hidden state
	t1 = [[0 for _ in range(T)] for _ in range(K)]
	#Table for storing paramaters for achieving T1's probability values
	t2 = [[0 for _ in range(T)] for _ in range(K+1)]

	#Calculate first states' probabilities from starting state
	for k in range(K):
		obs = backOffEmission(observed[0], emission_space, compound)
		obsInd = emission_space.index(obs)
		#P(State = state i) = P(Transition from starting state to state i) * P(State i emitting first observation)
		t1[k][0] = start_probs[k]*e_matrix[k][obsInd]

		t2[k][0] = 0

	#Second state: cannot use order 2 so do normally
	#Iterate through possible hidden states at time 1
	for j in range(K):
		temp1 = []
		temp2 = []
		#Iterate through possible transitions at time 0
		for k in range(K):

			stateMinus1Ind = state_space.index(state_space_singles[j])
			currentStateInd = state_space.index(state_space_singles[k])

			obs = backOffEmission(observed[1], emission_space, compound)

			obsInd = emission_space.index(obs)
			transitionProb = t_matrix[stateMinus1Ind][currentStateInd]
			emissionProb = e_matrix[currentStateInd][obsInd]


			#t1[k][0] = probability of path up to previous hidden state
			#t_matrix[k][j] = the transition probability from previous state (k) to current state(j)
			#e_matrix[j][observed[i]] = the emission probability to the observed state from current state (j)
			temp1.append(t1[j][0]*transitionProb*emissionProb)
			temp2.append(j)
		#Store best paths to each hidden state so far and their probabilities
		t1[j][1] = max(temp1)
		t2[j][1] = temp1.index(max(temp1))

	
	#Iterate through remaining observations
	for t in range(2, T):
		print(t)
		#Iterate through possible hidden states at time t-1
		for j in range(K):

			temp1 = []
			temp2 = []
			#Iterate through possible transitions from state at time t-1 to current state
			for k in range(K):

				#Find indices of last 2 states
				stateMinus1Ind = t2[k][t-1]
				stateMinus2Ind = t2[k][t-2]
				currentStateInd = k
				#Find states themselves
				currentState = state_space_singles[currentStateInd]
				stateMinus1 = state_space_singles[stateMinus1Ind]
				stateMinus2 = state_space_singles[stateMinus2Ind]

				slice_prob = sliceProb(stateMinus2, stateMinus1, currentState, observed, t, t_matrix, e_matrix, state_space_singles, state_space, emission_space, compound)

				#Calculate probability
				temp1.append(t1[j][t-1]*slice_prob)	
				temp2.append(k)
			#Store best paths to each hidden state so far and their probabilities
			t1[j][t] = max(temp1)
			t2[j][t] = temp1.index(max(temp1))

	ztemp = []
	for k in range(len(t1)):
		ztemp.append(t1[k][T-1])

	Z = [0 for i in range(T)]
	Z[-1] = ztemp.index(max(ztemp))

	X = [0 for i in range(T)]
	X[-1] = state_space_singles[Z[-1]]

	trange = reversed(range(T))

	for i in trange:
		Z[i-1] = t2[Z[i]][i]
		X[i-1] = state_space_singles[Z[i-1]]

	return X


def separateStates(state_space, N):
	#Seperate state space into lists of each size
	stateLists = []

	for n in reversed(range(N)):
		stateLists.append([])
		for item in state_space:
			if len(item)-1 == n:
				stateLists[-1].append(item)

	return stateLists


def genFirstN(t1, t2, obs, state_space, emission_space, start_probs, t_matrix, e_matrix, K, T, N):

	for t in range(1,N):
		print(t, 'of', T)

		#Iterate through possible hidden states at time t-1
		for j in range(K):

			temp1 = []
			temp2 = []
			#Iterate through possible transitions from state at time t-1
			for k in range(K):

				#Find indices of last states
				lastSeqInd = t2[k][:t]
				currentStateInd = t2[k][t]
				#Find states themselves
				currentState = state_space[currentStateInd]
				lastSeq = []
				for item in lastSeqInd:
					lastSeq.append(state_space[item])

				#Calculate probability
				temp1.append(no.sliceProb(lastSeq, currentState, obs[t], t_matrix, e_matrix, state_space, emission_space, N))
				temp2.append(k)
			#Store best paths to each hidden state so far and their probabilities
			t1[j][t] = max(temp1)
			t2[j][t] = temp1.index(max(temp1))

	return t1, t2

