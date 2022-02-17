import numpy as np
import os.path
import viterbi_bach as vb
import math
import evaluateChorales as ec
import harmIO
import helper as hp
import harm_skeleton as hs
import sys

##################################################################################
#		Version 1: Emission space harmony, State spaces each harmony note		 #
##################################################################################

def train_chord_skel_singleV1(dataset, part):
	#Harmonic symbols used
	symbols = []
	#Notes used
	notes = []

	#Transpose to key of C
	transposed = []
	for chorale in dataset:
		transposed.append(hp.transpose(chorale, 'C'))

	for chorale in transposed:
		symbols = hp.getSymbols(chorale, symbols)
		notes = hp.getSingles(chorale, part, notes)

	#Create empty matrices to store probabilities
	transition, emission, initial = vb.init_matrices(notes, symbols)

	#Update values in matrices
	for chorale in transposed:
		observed = hp.getLine(chorale, 'h')
		state_seq = hp.getLine(chorale, part)
		transition, emission, initial = vb.update_matrices(transition, emission, initial, notes, symbols, observed, state_seq)


	#Normalise so that all rows add to 1
	initial = hp.normalise(initial)
	emission = hp.normalise(emission)
	transition = hp.normalise(transition)


	return transition, emission, initial, notes


def train_chord_skelV1(dataset):

	transition_a, emission_a, initial_a, notes_a = train_chord_skel_singleV1(dataset, 'a')
	transition_t, emission_t, initial_t, notes_t = train_chord_skel_singleV1(dataset, 't')
	transition_b, emission_b, initial_b, notes_b = train_chord_skel_singleV1(dataset, 'b')

	return (transition_a, emission_a, initial_a, notes_a), \
		(transition_t, emission_t, initial_t, notes_t), \
		( transition_b, emission_b,initial_b, notes_b)

def generate_chordsV1(melody, harm_skel, trained):
	#Train model

	symbols = trained[2]


	alto = vb.viterbi(harm_skel, trained[1][1], symbols, hp.getTransition(trained, 'a'), hp.getEmission(trained, 'a'), hp.getInitial(trained, 'a'))
	tenor = vb.viterbi(harm_skel, trained[1][2], symbols, hp.getTransition(trained, 't'), hp.getEmission(trained, 't'), hp.getInitial(trained, 't'))
	bass = vb.viterbi(harm_skel, trained[1][3], symbols, hp.getTransition(trained, 'b'), hp.getEmission(trained, 'b'), hp.getInitial(trained, 'b'))


	return melody, hp.fill_skel(alto, melody), hp.fill_skel(tenor, melody), hp.fill_skel(bass, melody), hp.fill_skel( harm_skel, melody)


##################################################################################
#		Version 2: Emission space melody+harmony, State space ATB voicings		 #
##################################################################################

#States

def getStateSpaceSingleV2(voicings, state_space):
	for voicing in voicings:
		
		if voicing not in state_space:
			state_space.append(voicing)
	return state_space


def chordStateSeqV2(chorale):
	#Get the ATB voicing of the chord below the melody
	voicings = []

	a = hp.getLine(chorale, 'a')
	t = hp.getLine(chorale, 't')
	b = hp.getLine(chorale, 'b')

	a = hp.crotchets(a)
	t = hp.crotchets(t)
	b = hp.crotchets(b)

	T = len(a)

	for n in range(T):
		#Get ATB voicing of the chord

		voicing = (a[n], t[n], b[n])

		voicings.append(voicing)
	return voicings

def chordStateSpaceV2(dataset):
	state_space = []
	for chorale in dataset:

		state_seq = chordStateSeqV2(chorale)
		state_space = getStateSpaceSingleV2(state_seq, state_space)

	return state_space

#Emissions

def chordObsV2(chorale, version):
	#Get usable observation sequence for chord generation - zip together melody sections and harmonic symbols
	line = []
	melody = hp.getLine(chorale, 's')
	harmony = hp.getLine(chorale, 'h')

	if version == 2:
		melody_obs = hp.crotchets(melody)
	elif version == 3:
		melody_obs = hp.getCrotchetSections(melody)

	harmony_raw = hp.rawItems(harmony)
	T = len(harmony_raw)

	for t in range(T):
		current_h = harmony_raw[t]
		if version != 1:
			current_m = melody_obs[t]
			line.append((current_h,current_m))
		else:
			line.append(current_h)

	return line


def addChordEmissionV2(chorale, emission_space, version):
	#Function that returns list of tuples of melody notes with their respective harmony symbols

	observed = chordObsV2(chorale, version)

	for item in observed:
		if item not in emission_space:
			emission_space.append(item)
		if version != 1:
			if item[0] not in emission_space:
				emission_space.append(item[0])

	return emission_space

def chordEmissionSpaceV2(dataset, version):
	#Gather all pairings of a crotchet section of melody and harmonic symbol
	emission_space = []
	for chorale in dataset:
		
		emission_space = addChordEmissionV2(chorale, emission_space, version)

	return emission_space


def train_chord_skelV2(dataset, version, verbose=True):	

	# #Soprano note + harmony symbol combinations used
	emission_space = chordEmissionSpaceV2(dataset, version)
	#ATB voicings used
	state_space = chordStateSpaceV2(dataset)

	#Create empty matrices to store probabilities
	transition, emission, initial = vb.init_matrices(state_space, emission_space)


	#Update values in matrices
	for chorale in dataset:
		if verbose==True:
			print('Training chords from chorale', dataset.index(chorale))


		if version == 1:
			observed = hp.rawItems(hp.getLine(chorale,'h'))
		else:
			observed = chordObsV2(chorale, version)


		state_seq = chordStateSeqV2(chorale)


		transition, emission, initial = vb.update_matrices(transition, emission, initial, state_space, emission_space, observed, state_seq)

	print('Normalising transition')
	transition = hp.normalise(transition)
	print('Normalising emission')
	emission = hp.normalise(emission)
	print('Normalising initial')
	initial = hp.normalise(initial)

	return transition, emission, initial, state_space, emission_space


def line_harmony(line, harm):
	#Function to create a usable melody/harmony list for generating chord skeleton
	harm_raw = hp.rawItems(harm)

	line = hp.crotchets(line, acceptMinims=True)

	new = []
	
	T = len(line)

	for t in range(T):
		current_l = line[t]
		current_h = harm_raw[t]
		new.append((hp.raw(current_h),hp.raw(current_l)))

	return new

def sections_harmony(melody, harm_skel):
	sections = hp.getCrotchetSections(melody)
	
	new = []

	T = len(harm_skel)

	for t in range(T):
		new.append((harm_skel[t], sections[t]))

	return new

def generate_chordsV2(melody, harm_skel, trained_chords, version, verbose=False):

	if version == 3:
		observed = sections_harmony(melody, harm_skel)
	elif version == 2:
		observed = line_harmony(melody, harm_skel)
	elif version == 1:
		observed = harm_skel

	transition, emission, initial, state_space, emission_space = trained_chords

	if version == 1:
		chord_skel = vb.viterbi(observed, state_space, emission_space, transition, emission, initial, verbose)
	else:
		chord_skel = vb.viterbi(observed, state_space, emission_space, transition, emission, initial, verbose, compound=True)


	return chord_skel


def formatChordSkel(melody, chord_skel, harm_skel, chorale=None, key='C'):
	#Turn interval-based representation into actual notes

	melody_sections = hp.getCrotchetSections(melody)
	melody_crotchets = hp.crotchets(melody)
	T = len(melody_crotchets)


	soprano = []
	alto = []
	tenor = []
	bass = []
	harm_new = []

	notes = hp.allNotes()

	for t in range(T):
		chord_temp = chord_skel[t]

		sop_crotchet = melody_crotchets[t]

		held = False

		if t>0:
			#See if melody is continuing a 2+ beat note
			if hp.allDashed(melody_crotchets[t]) and hp.allDashed(melody_crotchets[t-1][1:]):
				held=True

		temp_a = chord_temp[0]
		temp_t = chord_temp[1]
		temp_b = chord_temp[2]

		if held == True:
			last_a = chord_skel[t-1][0]
			last_t = chord_skel[t-1][1]
			last_b = chord_skel[t-1][2]

			if temp_a == hp.raw(last_a):
				alto.append(hp.dashed(temp_a))
			if temp_t == hp.raw(last_t):
				tenor.append(hp.dashed(temp_t))
			if temp_b == hp.raw(last_b):
				bass.append(hp.dashed(temp_b))
		else:
			alto.append(temp_a)
			tenor.append(temp_t)
			bass.append(temp_b)
		harm_new.append(harm_skel[t])


		for i in range(1,4):
			
			note = melody[t+i]


			#Fill out lines
			soprano.append(note)
			alto.append(hp.dashed(temp_a))
			tenor.append(hp.dashed(temp_t))
			bass.append(hp.dashed(temp_b))
			harm_new.append(hp.dashed(harm_skel[t]))

		
	if chorale == None:
		beats_m = [4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3]

		newChorale = [key,'-', [], beats_m, melody, alto, tenor, bass, harm_new]
	else:
		newChorale = [key,chorale[1], chorale[2], chorale[3], melody, alto, tenor, bass, harm_new]
	return newChorale

if __name__ == "__main__":

	# try:
	# 	choraleNo = input('Please enter the dataset index of the melody you wish to harmonise')
	# except:
	# 	print('Incorrect user input, defaulting to minor chorale 383')

	#Extract test chorale data
	try:
		chorale_m = hp.transpose(harmIO.extract(41), 'C')
	except:
		print('Error in extracting chorale')
		sys.exit()
	melody_m = hp.getLine(chorale_m, 's')
	beats_m = hs.getBeat(chorale_m)
	tonality = chorale_m[0]

	# melody_m = ['F#1','F#1-','F#1-','F#1-','H 1', 'H 1-','H 1-','H 1-','H 1-','H 1-','H 1-','H 1-','H 1-','H 1-','H 1-','H 1-', \
	# 			'C#2','C#2-','C#2-','C#2-', 'D 2','D 2-','D 2-','D 2-','D 2-','D 2-','D 2-','D 2-','D 2-','D 2-','D 2-','D 2-', \
	# 			'H 1', 'H 1-','H 1-','H 1-','D 2','D 2-','D 2-','D 2-','D 2','D 2-','D 2-','D 2-','D 2','D 2-','D 2-','D 2-', \
	# 			'E 2','E 2-','E 2-','E 2-','F#2','F#2-','F#2-','F#2-','F#2-','F#2-','F#2-','F#2-','F#2-','F#2-','F#2-','F#2-', \
	# 			'D 2','D 2-','D 2-','D 2-','D 2','D 2-','D 2-','D 2-','C#2','C#2-','H 1', 'H 1-','A#1','A#1-','A#1-','A#1-', \
	# 			'A#1','A#1-','A#1-','A#1-','H 1', 'H 1-','H 1-','H 1-', 'A 1','A 1-','A 1-','A 1-', 'G 1','G 1-','G 1-','G 1-', \
	# 			'A 1','A 1-','A 1-','A 1-','F#1','F#1-','F#1-','F#1-','F#1-','F#1-','F#1-','F#1-', 'D 1','D 1-','D 1-','D 1-', \
	# 			'E 1','E 1-','E 1-','E 1-','F#1','F#1-','F#1-','F#1-','F#1-','F#1-','F#1-','F#1-','F#1','F#1-','F#1-','F#1-']

	# beats_m = [4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3]

	#Gather dataset
	dataset_m, _ = harmIO.data('All', '-')
	#Transpose to same key
	dataset_m = hp.transposeData(dataset_m)
	#Train harmony
	trained_harm_m = hs.train_harm_skel(dataset_m, 3)
	#Gnerate harmony
	harm_skel_m = hs.generate_harm(melody_m, trained_harm_m, 3, beats_m)
	#Train chords
	trained_chords_m = train_chord_skelV2(dataset_m, 1)
	#Generate chords
	chord_skel_m = generate_chordsV2(melody_m, harm_skel_m, trained_chords_m, 1)
	#Format
	formatted_m = formatChordSkel(melody_m, chord_skel_m, harm_skel_m, chorale_m)
	print('Minor chorale: 383')
	#Output
	harmIO.printChorale(formatted_m )
