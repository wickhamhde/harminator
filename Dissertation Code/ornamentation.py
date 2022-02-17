import numpy as np
import os.path
import viterbi_bach as vb
import math
import harmIO
import helper as hp
import harm_skeleton as hs
import chord_skeleton as cs

#####################################################################################################
#								COMMON FUNCTIONS													#
#####################################################################################################


def removeSemiquavers(item):
	new = []

	new.append(item[0])	
	new.append(item[0])
	new.append(item[2])
	new.append(item[2])

	return new

def semiquaverIntervals(item, semiquavers=True):
	#Convert note representation to interval only
	if semiquavers == False:
		item = removeSemiquavers(item)
	new = []
	first = hp.raw(item[0])
	new.append(0)
	for i in range(1,4):
		new.append(hp.interval(first, hp.raw(item[i]), False))

	new = tuple(new)

	return new


def removeUnseen(emission_space, observed):
	#Weed out unseen observations so they don't interfere with viterbi
	new = []
	removedInds = []

	T = len(observed)

	for t in range(T):
		if observed[t] not in emission_space:
			removedInds.append(t)
		else:
			new.append(observed[t])

	return new, removedInds

#####################################################################################################
#								ORNAMENTATION VERSION 1												#
#####################################################################################################

def ornamentationObsV1(line, harmony, reverse=False):
	
	line = hp.crotchets(line)
	harmony_raw = hp.rawItems(harmony)

	observed = []
	T = len(line)

	for t in range(T):
		observed.append((line[t], harmony_raw[t]))

	if reverse == True:
		observed_reversed = []
		for i in reversed(observed):
			observed_reversed.append(i)

		return observed_reversed
	else:
		return observed

def ornamentationEmissionSpaceV1(dataset, voicePart, emission_space):
	
	for chorale in dataset:
		line = hp.getLine(chorale, voicePart)
		harmony = hp.getLine(chorale, 'h')
		temp = ornamentationObsV1(line, harmony)

		for item in temp:
			if item not in emission_space:
				emission_space.append(item)
			if item[0] not in emission_space:
				emission_space.append(item[0])

	return emission_space


def ornamentationStateSeqV1(line, semiquavers, reverse=False):

	line_sections = hp.getCrotchetSections(line)

	state_seq = []

	if reverse==True:
		for item in reversed(line_sections):
			state_seq.append(semiquaverIntervals(item, semiquavers))
		return state_seq
	else:
		for item in line_sections:
			state_seq.append(semiquaverIntervals(item, semiquavers))
		return state_seq	
	
def ornamentationStateSpaceV1(dataset, voicePart, state_space, semiquavers, reverse=False):

	for chorale in dataset:
		line = hp.getLine(chorale, voicePart)

		temp = ornamentationStateSeqV1(line, semiquavers, reverse)

		for item in temp:
			if item not in state_space:
				state_space.append(item)

	return state_space

def trainOrnamentationSingleV1(dataset, voicePart, emission_space, state_space,semiquavers, reverse):

	transition, emission, initial = vb.init_matrices(state_space, emission_space)

	for chorale in dataset:
		line = hp.getLine(chorale, voicePart)
		harmony = hp.getLine(chorale, 'h')

		state_seq = ornamentationStateSeqV1(line, semiquavers, reverse)
		observed = ornamentationObsV1(line, harmony, reverse)

		transition, emission, initial = vb.update_matrices(transition, emission, initial, state_space, emission_space, observed, state_seq)

	try:
		ind1 = emission_space.index('C 1')
		ind2 = state_space.index('H 0', (0,0,0,0))
		print(emission[ind2][ind1])
	except:
		print(0)

	print('Normalising transition')
	transition = hp.normalise(transition)
	print('Normalising emission')
	emission = hp.normalise(emission)
	print('Normalising initial')
	initial = hp.normalise(initial)

	return transition, emission, initial


def trainOrnamentationV1(dataset, reverse, semiquavers=True):

	emission_space_a = ornamentationEmissionSpaceV1(dataset, 'a', [])
	emission_space_t = ornamentationEmissionSpaceV1(dataset, 't', [])
	emission_space_b = ornamentationEmissionSpaceV1(dataset, 'b', [])


	state_space_a = ornamentationStateSpaceV1(dataset, 'a', [], semiquavers)
	state_space_t = ornamentationStateSpaceV1(dataset, 't', [], semiquavers)
	state_space_b = ornamentationStateSpaceV1(dataset, 'b', [], semiquavers)


	transition_a, emission_a, initial_a = trainOrnamentationSingleV1(dataset, 'a', emission_space_a, state_space_a, semiquavers, reverse)
	transition_t, emission_t, initial_t = trainOrnamentationSingleV1(dataset, 't', emission_space_t, state_space_t, semiquavers, reverse)
	transition_b, emission_b, initial_b = trainOrnamentationSingleV1(dataset, 'b', emission_space_b, state_space_b, semiquavers, reverse)

	return  [transition_a, emission_a, initial_a, state_space_a, emission_space_a], \
			[transition_t, emission_t, initial_t, state_space_t, emission_space_t], \
			[transition_b, emission_b, initial_b, state_space_b, emission_space_b] \

def generate_ornamentation_singleV1(generatedChorale, voicePart, trained_orn):
	#Generte single line of ornamentation using viterbi algorithm
	transition = trained_orn[0]
	emission = trained_orn[1]
	initial = trained_orn[2]
	state_space = trained_orn[3]
	emission_space = trained_orn[4]

	print(state_space)

	
	line = hp.getLine(generatedChorale, voicePart)
	harmony = hp.getLine(generatedChorale, 'h')

	observed = ornamentationObsV1(line, harmony)
	obsInds = []

	orn = vb.viterbi(observed, state_space, emission_space, transition, emission, initial, orn=True, compound=True)

	return orn


def generate_ornamentationV1(generatedChorale, trained_orn, reverse):

	orn_a = generate_ornamentation_singleV1(generatedChorale, 'a', trained_orn[0])
	
	orn_t = generate_ornamentation_singleV1(generatedChorale, 't', trained_orn[1])
	
	orn_b= generate_ornamentation_singleV1(generatedChorale, 'b', trained_orn[2])

	orn_a_final = []
	orn_t_final = []
	orn_b_final = []

	if reverse==True:
		for item in reversed(orn_a):
			orn_a_final.append(item)
		for item in reversed(orn_t):
			orn_t_final.append(item)
		for item in reversed(orn_b):
			orn_b_final.append(item)

		return orn_a_final, orn_t_final, orn_b_final

	else:
		return (orn_a, orn_t, orn_b)


def formatOrnamentation(chorale, orn_a, orn_t, orn_b):

	notes = hp.allNotes()

	alto_old = hp.crotchets(hp.getLine(chorale, 'a'))
	tenor_old = hp.crotchets(hp.getLine(chorale, 't'))
	bass_old = hp.crotchets(hp.getLine(chorale, 'b'))

	print(alto_old)
	print(tenor_old)
	print(bass_old)

	alto = []
	tenor = []
	bass = []

	T = len(orn_a)

	for t in range(T):

		#Convert interval representation to notes
		first_a = alto_old[t]
		ints_a = orn_a[t]

		first_t = tenor_old[t]
		ints_t = orn_t[t]

		first_b = bass_old[t]
		ints_b = orn_b[t]

		#Temporary chunk for this beat so that crotchets are separated
		alto_chunk = []
		tenor_chunk = []
		bass_chunk = []


		for i in range(4):
			note_a = hp.transpose_single(ints_a[i], first_a, notes)
			note_t = hp.transpose_single(ints_t[i], first_t, notes)
			note_b = hp.transpose_single(ints_b[i], first_b, notes)

			if i != 0:
				if hp.raw(alto_chunk[-1]) == note_a:
					note_a = hp.dashed(note_a)
				if hp.raw(tenor_chunk[-1]) == note_t:
					note_t = hp.dashed(note_t)
				if hp.raw(bass_chunk[-1]) == note_b:
					note_b = hp.dashed(note_b)

			alto_chunk.append(note_a)
			tenor_chunk.append(note_t)
			bass_chunk.append(note_b)

		for i in range(4):

			alto.append(alto_chunk[i])
			tenor.append(tenor_chunk[i])
			bass.append(bass_chunk[i])

	chorale[5] = alto
	chorale[6] = tenor
	chorale[7] = bass

	return chorale


#####################################################################################################
#								ORNAMENTATION VERSION 2												#
#####################################################################################################

def trainOrnamentation_single2(dataset, voicePart, semiquavers):
	#Alternative ornamentation method using context size 2 and standard markov model

	#Possible types of ornamentation
	state_space = []
	#Contexts of size 2 (current crotchet, harmony)
	context_space_2 = []
	#Context of size 3 (current crotchet, harmony, next crotchet)
	context_space_3 = []

	for chorale in dataset:
		line = hp.getLine(chorale, voicePart)
		harmony = hp.getLine(chorale, 'h')

		line_crotchets = hp.crotchets(line)
		sections = hp.getCrotchetSections(line)
		harm_skel = hp.raw(harmony)

		T = len(line_crotchets)
		for t in range(T):

			#Populate state space with interval representation for ornamentation
			intervals = semiquaverIntervals(sections[t], semiquavers)
			if intervals not in state_space:
				state_space.append(intervals)

			#Populate context size 2 space
			context2 = (line_crotchets[t], harm_skel[t])
			if context2 not in context_space_2:
				context_space_2.append(context2)

			#Populate context size 3 space
			if t < T-1: #Prevent index error
				context3 = (line_crotchets[t], harm_skel[t], line_crotchets[t+1])
				if context3 not in context_space_3:
					context_space_3.append(context3)


	#Initialise probability matrices
	matrix_context_2 = np.zeros((len(context_space_2), len(state_space)))
	matrix_context_3 = np.zeros((len(context_space_3), len(state_space)))

	for chorale in dataset:
		line = hp.getLine(chorale, voicePart)
		harmony = hp.getLine(chorale, 'h')

		line_crotchets = hp.crotchets(line)
		sections = hp.getCrotchetSections(line)
		harm_skel = hp.raw(harmony)

		T = len(line_crotchets)
		for t in range(T):

			intervals = semiquaverIntervals(sections[t], semiquavers)
			context2 = (line_crotchets[t], harm_skel[t])

			#Increment probability matrices
			intervalsInd = state_space.index(intervals)
			context2Ind = context_space_2.index(context2)
			matrix_context_2[context2Ind][intervalsInd] += 1

			if t < T-1: #Prevent index error
				context3 = (line_crotchets[t], harm_skel[t], line_crotchets[t+1])
				context3Ind = context_space_3.index(context3)
				matrix_context_3[context3Ind][intervalsInd] += 1

	print('Normalising context 2')
	hp.normalise(matrix_context_2)
	print('Normalising context 3')
	hp.normalise(matrix_context_3)

	return matrix_context_2, matrix_context_3, state_space, context_space_2, context_space_3

def trainOrnamentation2(dataset, semiquavers):
	#Train ornamentation for all parts

	trained_orn_a = trainOrnamentation_single2(dataset, 'a', semiquavers)
	trained_orn_t = trainOrnamentation_single2(dataset, 't', semiquavers)
	trained_orn_b = trainOrnamentation_single2(dataset, 'b', semiquavers)

	return trained_orn_a, trained_orn_t, trained_orn_b

def generate_ornamentation_single2(line, harm_skel, trained_orn, semiquavers):
	#Generate single line of ornamentation

	#Extract training data
	matrix_context_2 = trained_orn[0]
	matrix_context_3 = trained_orn[1]
	state_space = trained_orn[2]
	context_space_2 = trained_orn[3]
	context_space_3 = trained_orn[4]

	line_crotchets = hp.crotchets(line)
	ornamented = []

	T = len(line_crotchets)
	for t in range(T):
		#Attempt with context 3 but back off if necessary
		try:
			context3 = (line_crotchets[t], harm_skel[t], line_crotchets[t+1])
			context3Ind = context_space_3.index(context3)
			#Find most likely ornamentation
			bestProb = max(matrix_context_3[context3Ind])
			bestInd = matrix_context_3[context3Ind].tolist().index(bestProb)
			best = state_space[bestInd]
			ornamented.append(best)
		#Either context not in space or have reached last item of sequence
		except:
			try:
				context2 = (line_crotchets[t], harm_skel[t])
				context2Ind = context_space_2.index(context2)
				#Find most likely ornamentation
				bestProb = max(matrix_context_2[context2Ind])
				bestInd = matrix_context_2[context2Ind].tolist().index(bestProb)
				best = state_space[bestInd]
				ornamented.append(best)
			#Default to no ornamentation
			except:
				ornamented.append((0,0,0,0))

	return ornamented

def generate_ornamentation2(generated, trained_orn, semiquavers=False):
	#Generate ornamentation for whole chord skeleton

	trained_orn_a = trained_orn[0]
	trained_orn_t = trained_orn[1]
	trained_orn_b = trained_orn[2]

	alto = hp.getLine(generated, 'a')
	tenor = hp.getLine(generated, 't')
	bass = hp.getLine(generated, 'b')
	harm_skel = hp.getLine(generated, 'h')

	orn_a = generate_ornamentation_single2(alto, harm_skel, trained_orn_a, semiquavers)
	orn_t = generate_ornamentation_single2(tenor, harm_skel, trained_orn_t, semiquavers)
	orn_b = generate_ornamentation_single2(bass, harm_skel, trained_orn_b, semiquavers)

	return (orn_a, orn_t, orn_b)


#####################################################################################################
#								ORNAMENTATION VERSION 3												#
#####################################################################################################

def ornSpaces3(chorale, state_space, emission_space):
	#State space: ornamented segments in ATB
	#Emission space: initial crotchets of SATB
	soprano_crotchets = hp.crotchets(hp.getLine(chorale, 's'))
	alto_crotchets = hp.crotchets(hp.getLine(chorale, 'a'))
	tenor_crotchets = hp.crotchets(hp.getLine(chorale, 't'))
	bass_crotchets = hp.crotchets(hp.getLine(chorale, 'b'))

	alto_sections = hp.getCrotchetSections(hp.getLine(chorale, 'a'))
	tenor_sections = hp.getCrotchetSections(hp.getLine(chorale, 't'))
	bass_sections = hp.getCrotchetSections(hp.getLine(chorale, 'b'))


	T = len(alto_crotchets)

	for t in range(T):
		if (soprano_crotchets[t], alto_crotchets[t], tenor_crotchets[t], bass_crotchets[t]) not in emission_space:
			emission_space.append((soprano_crotchets[t], alto_crotchets[t], tenor_crotchets[t], bass_crotchets[t]))

		item = (semiquaverIntervals(alto_sections[t]), semiquaverIntervals(tenor_sections[t]), semiquaverIntervals(bass_sections[t]))
		if item not in state_space:
			state_space.append(item)

	return state_space, emission_space

def ornObsStateSeq3(chorale):
	observed = []
	state_seq = []


	soprano_crotchets = hp.crotchets(hp.getLine(chorale, 's'))
	alto_crotchets = hp.crotchets(hp.getLine(chorale, 'a'))
	tenor_crotchets = hp.crotchets(hp.getLine(chorale, 't'))
	bass_crotchets = hp.crotchets(hp.getLine(chorale, 'b'))

	alto_sections = hp.getCrotchetSections(hp.getLine(chorale, 'a'))
	tenor_sections = hp.getCrotchetSections(hp.getLine(chorale, 't'))
	bass_sections = hp.getCrotchetSections(hp.getLine(chorale, 'b'))


	T = len(alto_crotchets)
	for t in range(T):
		observed.append((soprano_crotchets[t], alto_crotchets[t], tenor_crotchets[t], bass_crotchets[t]))
		item = (semiquaverIntervals(alto_sections[t]), semiquaverIntervals(tenor_sections[t]), semiquaverIntervals(bass_sections[t]))
		state_seq.append(item)

	return state_seq, observed

def trainOrnamentation3(dataset):
	#Alternative method for ornamentation - probabilistic but not Markov Model

	state_space = []
	emission_space = []

	for chorale in dataset:
		state_space, emission_space = ornSpaces3(chorale, state_space, emission_space)

	transition, emission, initial = vb.init_matrices(state_space, emission_space)

	for chorale in dataset:
		state_seq, observed = ornObsStateSeq3(chorale)

		transition, emission, initial = vb.update_matrices(transition, emission, initial, state_space, emission_space, observed, state_seq)
		
	return (transition, emission, initial, state_space, emission_space)

def generate_ornamentation_3(generated, trained_orn):
	#Go through the chorale and find most appropriate ornamentation
	_, observed = ornObsStateSeq3(generated)

	transition = trained_orn[0]
	emission = trained_orn[1]
	initial = trained_orn[2]
	state_space = trained_orn[3]
	emission_space = trained_orn[4]

	orn = vb.viterbi(observed, state_space, emission_space, transition, emission, initial)

	return orn


if __name__ == "__main__":
	dataset, _ = harmIO.data('All', '-')
	dataset = hp.transposeData(dataset)

	chorale = harmIO.extract(383)
	chorale = hp.transpose(chorale, 'C')


	melody = hp.getLine(chorale, 's')
	harmony = hp.rawItems(hp.getLine(chorale, 'h'))
	beats = hs.getBeat(chorale)

	trained_harm = hs.train_harm_skel(dataset,3)
	harm_skel = hs.generate_harm(melody, trained_harm, 3, beats)

	trained_chords = cs.train_chord_skelV2(dataset, 3)
	chord_skel = cs.generate_chordsV2(melody, harm_skel, trained_chords, 1)
	formatted = cs.formatChordSkel(melody, chord_skel, harm_skel, chorale)
	harmIO.printChorale(formatted)

	trained_orn = trainOrnamentation3(dataset)
	orn = generate_ornamentation_3(formatted, trained_orn)

	print(orn)
