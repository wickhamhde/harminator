import numpy as np
import os.path
import viterbi_bach as vb
import math
import evaluateChorales as ec
import harmIO
import helper as hp
import chord_skeleton as cs
'''
File for dealing with the training and generation of the harmonic skeleton
'''

##################################################################################
#							Version 1: Single notes as emissions 				 #
##################################################################################

def generate_harm_v1(melody, trained):

	symbols = trained[3]
	notes = hp.getItemsSingle([], melody)
	#print(transition)

	transition = trained[0]
	emission = trained[1]
	initial = trained[2]

	melody = hp.crotchets(melody)
	harm_skel = vb.viterbi(melody, symbols, notes, transition, emission, initial)

	return harm_skel

##################################################################################
#						Version 2: Ornamented beats as emissions				 #
##################################################################################

def harmEmissionSpaceV2(chorale, emission_space):
	#Build the emission space, consisting of crotchet-length sections of the melody

	melody = hp.getLine(chorale, 's')

	observed = getHarmObsV2(melody)


	for item in observed:
		if item not in emission_space:
			emission_space.append(item)

	return emission_space


def getHarmObsV2(melody):
	#Get observed sequence for harmonic skeleton
	observed = []
	melody_sections = hp.getCrotchetSections(melody)

	T = len(melody_sections)

	for t in range(T):	
		observed.append(melody_sections[t])


	return observed

def harmStateSeqV2(harmony):
	#Get state sequence for use in harmonic skeleton

	harm_raw = hp.rawItems(harmony)

	new = []

	T = len(harm_raw)

	for t in range(T):	
		new.append(harm_raw[t])

	return new

##################################################################################
#			Version 3: Ornamented beats and beat numbers as emissions			 #
##################################################################################

def getBeat(chorale):
	#Get a list of the beat of the bar for use with harmonic skeleton

	beats = []
	beatCount = 1
	barNumbers = chorale[3]

	#Vlue to check for a potential anacrusis
	hadFirstBar = False


	T = int(len(barNumbers)/4)

	for t in range(T):
		time = t*4

		beat = barNumbers[time]
		#True of the current step being examined isn't first of bar, False otherwise
		notFirstOfBar = hp.isDashed(beat)
		if notFirstOfBar:
			if hadFirstBar:
				beatCount += 1
				beats.append(beatCount)
			else:
				beats.append(4)
		else:
			if not hadFirstBar:
				hadFirstBar = True
			beatCount = 1
			beats.append(beatCount)

	return beats

#VERSION USING BEAT OF BAR
def harmEmissionSpaceV3(chorale, emission_space):
	#Build the emission space, consisting of crotchet-length sections of the melody

	melody = hp.getLine(chorale, 's')
	beats = getBeat(chorale)

	melody_sections = hp.getCrotchetSections(melody)


	T = len(melody_sections)

	for t in range(T):
		item = (melody_sections[t], beats[t])
		if item not in emission_space:
			emission_space.append(item)

	return emission_space

#VERSION USING BEAT OF BAR
def getHarmObsV3(melody, beats):
	#Get observed sequence for harmonic skeleton
	observed = []
	melody_sections = hp.getCrotchetSections(melody)

	T = len(melody_sections)

	for t in range(T):
		observed.append((melody_sections[t], beats[t]))

	return observed


##################################################################################
#					Version 4: Order 2, emission with beats and notes			 #
##################################################################################

def getHarmStateSpaceV4(chorale, state_space):
	#Get order 2 state space for harmonic skeleton

	#First sort order 1 state (single symbols)
	state_space = hp.getSymbols(chorale, state_space)

	#Now sort order 2
	symbols = hp.getLine(chorale, 'h')
	symbols_raw = hp.rawItems(symbols)

	T = len(symbols_raw)

	for t in range(T-1):
		if (symbols_raw[t], symbols_raw[t+1]) not in state_space:
			state_space.append((symbols_raw[t], symbols_raw[t+1]))

	return state_space


##################################################################################
#						Training functions for all orders						 #
##################################################################################

def train_harm_skel_v1(dataset):
	#Train harmonic skeleton using first order markov system
	notes = []
	symbols = []

	for chorale in transposed:
		#Gather possible notes
		notes = hp.getSingles(chorale, 's', notes)
		#Gather possible symbols
		symbols = hp.getSymbols(chorale, symbols)

	#Create empty matrices to store probabilities
	transition, emission, initial = init_matrices(symbols, notes)

	#Update values in matrices
	for chorale in transposed:
		observed = hp.getLine(chorale, 's')
		state_seq = hp.getLine(chorale, 'h')
		transition, emission, initial = vb.update_matrices(transition, emission, initial, symbols, notes, observed, state_seq)

	#Normalise so that all rows add to 1
	initial = hp.normalise(initial)
	emission = hp.normalise(emission)
	transition = hp.normalise(transition)

	return transition, emission, initial, symbols, notes

def train_harm_skel(dataset, version, verbose=True):
	#Train harmonic skeleton using markov system


	emission_space = []
	state_space = []

	for chorale in dataset:

		#Gather emission space
		if version == 1:
			emission_space = hp.getSingles(chorale, 's', emission_space)
		elif version == 2:
			emission_space = harmEmissionSpaceV2(chorale, emission_space)
		elif version == 3:
			emission_space = harmEmissionSpaceV3(chorale, emission_space)

		#Gather possible symbols
		state_space = hp.getSymbols(chorale, state_space)


	#Create empty matrices to store probabilities
	transition, emission, initial = vb.init_matrices(state_space, emission_space)

	#Update values in matrices
	for chorale in dataset:
		if verbose==True:
			print('Training harmony from chorale', str(dataset.index(chorale)+1))

		melody = hp.getLine(chorale, 's')
		if version == 1:
			observed = melody
		elif version == 2:
			observed = getHarmObsV2(melody)
		elif version == 3:
			beats = getBeat(chorale)
			observed = getHarmObsV3(melody, beats)

		harmony = hp.getLine(chorale, 'h')
		if version != 1:		
			state_seq = harmStateSeqV2(harmony)
		else:
			state_seq = harmony

		transition, emission, initial = vb.update_matrices(transition, emission, initial, state_space, emission_space, observed, state_seq)

	print('Normalising transition')
	transition = hp.normalise(transition)
	print('Normalising emission')
	emission = hp.normalise(emission)
	print('Normalising initial')
	initial = hp.normalise(initial)

	return transition, emission, initial, state_space, emission_space


def train_harm_skel_order_2(dataset, version=4, verbose=True):
	#Train harmonic skeleton using markov system


	emission_space = []
	state_space = []

	for chorale in dataset:

		#Gather possible notes
		if version == 4:
			emission_space = harmEmissionSpaceV2(chorale, emission_space)
		elif version == 5:
			emission_space = harmEmissionSpaceV3(chorale, emission_space)

		#Gather possible symbols
		state_space = getHarmStateSpaceV4(chorale, state_space)


	#Create empty matrices to store probabilities
	transition, emission, initial = vb.init_matrices(state_space, emission_space)

	#Update values in matrices
	for chorale in dataset:
		if verbose==True:
			print('Training harmony from chorale', str(dataset.index(chorale)+1))

		melody = hp.getLine(chorale, 's')
	
		if version == 4:
			observed = getHarmObsV2(melody)
		elif version == 5:
			beats = getBeat(chorale)
			observed = getHarmObsV3(melody, beats)

		harmony = hp.getLine(chorale, 'h')		
		state_seq = hp.rawItems(harmony)

		transition, emission, initial = vb.update_matrices_order2(transition, emission, initial, state_space, emission_space, observed, state_seq)

	print('Normalising transition')
	transition = hp.normalise(transition)
	print('Normalising emission')
	emission = hp.normalise(emission)
	print('Normalising initial')
	initial = hp.normalise(initial)

	return transition, emission, initial, state_space, emission_space

##################################################################################
#					Generation functions for all orders							 #
##################################################################################

def generate_harm(melody, trained_harm, version, beats=None):

	if version == 1:
		observed = hp.crotchets(melody)
	elif version == 2:
		observed = getHarmObsV2(melody)
	elif version == 3:

		observed = getHarmObsV3(melody, beats)

	transition, emission, initial, state_space, emission_space = trained_harm


	if version != 3 and version != 5:
		harm_skel = vb.viterbi(observed, state_space, emission_space, transition, emission, initial)
	else:
		harm_skel = vb.viterbi(observed, state_space, emission_space, transition, emission, initial, compound=True)


	return harm_skel

def generate_harm_order_2(melody, trained_harm, version=4, beats=None):

	if version == 5:
		observed = getHarmObsV3(melody, beats)
		compound = True
	else:
		observed = getHarmObsV2(melody)
		compound = False

	transition, emission, initial, state_space, emission_space = trained_harm


	harm_skel = vb.viterbi_order_2(observed, state_space, emission_space, transition, emission, initial, compound)


	return harm_skel


##################################################################################
#								Main Function									 #
##################################################################################

if __name__ == "__main__":
	dataset, _ = harmIO.data('All', '-')
	
	dataset = hp.transposeData(dataset)

	chorale_original = harmIO.extract(383)
	
	chorale = hp.transpose(chorale_original, 'C')

	melody = hp.getLine(chorale, 's')
	harmony = hp.rawItems(hp.getLine(chorale, 'h'))
	beats = getBeat(chorale)

	trained_harm = train_harm_skel_order_2(dataset, 5)
	harm_skel = generate_harm_order_2(melody, trained_harm, 5, beats)

	print(harm_skel)
	print(len(harmony), len(harm_skel))

	# trained_chords = cs.train_chord_skel(dataset)

	# chord_skel = cs.generate_chords(melody, harm_skel, trained_chords)
	# print(chord_skel)
	# formatted = cs.formatChordSkel(melody, chord_skel, harm_skel, chorale)

	# harmIO.printChorale(formatted)


	