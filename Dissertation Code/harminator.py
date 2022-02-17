import numpy as np
import os.path
import viterbi_bach as vb
import math
import evaluateChorales as ec
import harmIO

import helper as hp

	
def train(dataset):
	transition_h, emission_h, initial_h, symbols, notes_h = train_harm_skel(dataset)
	(transition_a, emission_a, initial_a, notes_a),(transition_t, emission_t, initial_t, notes_t),( transition_b, emission_b,initial_b, notes_b) = train_chord_skel(dataset)

	matrices = ((transition_h, emission_h, initial_h), (transition_a, emission_a, initial_a),(transition_t, emission_t, initial_t),( transition_b, emission_b,initial_b))
	notes = (notes_h, notes_a, notes_t, notes_b)


	return matrices, notes, symbols


def generate_chords(melody, harm_skel, trained):
	#Train model

	symbols = trained[2]


	alto = vb.viterbi(harm_skel, trained[1][1], symbols, getTransition(trained, 'a'), getEmission(trained, 'a'), getInitial(trained, 'a'))
	tenor = vb.viterbi(harm_skel, trained[1][2], symbols, getTransition(trained, 't'), getEmission(trained, 't'), getInitial(trained, 't'))
	bass = vb.viterbi(harm_skel, trained[1][3], symbols, getTransition(trained, 'b'), getEmission(trained, 'b'), getInitial(trained, 'b'))


	return melody, fill_skel(alto, melody), fill_skel(tenor, melody), fill_skel(bass, melody),fill_skel( harm_skel, melody)

def ornamentationStateSeq(line):

	line_sections = mlou.getCrotchetSections(line)
	crotchets = mlou.crotchets(line)
	new = []

	for t in range(len(line_sections)):
		temp = []
		for i in range(4):
			temp.append(ec.interval(crotchets[t], line_sections[t][i], False))
		new.append(tuple(temp))

	return new

def ornamentationObs(line):
	crotchets = mlou.crotchets(line)

	return crotchets

def trainOrnamentationSingle(dataset, voicePart):
	state_space = []
	emission_space = []

	for chorale in dataset:
		line = getLine(chorale, voicePart)
		state_seq = ornamentationStateSeq(line)
		observed = ornamentationObs(line)

		for item in state_seq:
			if item not in state_space:
				state_space.append(item)
		for item in observed:
			if item not in emission_space:
				emission_space.append(item)

	transition, emission, initial = init_matrices(state_space, emission_space)

	#Update values in matrices
	for chorale in dataset:
		line = getLine(chorale, voicePart)
		observed = ornamentationObs(line)
		state_seq = ornamentationStateSeq(line)

		for i in range(len(observed)):
			if observed[i] != state_seq[i][0]:
				print(observed[i], state_seq[i])
		transition, emission, initial = update_matrices(transition, emission, initial, state_space, emission_space, observed, state_seq)


	#Normalise so that all rows add to 1
	initial = normalise(initial)
	emission = normalise(emission)
	transition = normalise(transition)


	return transition, emission, initial, state_space, emission_space

def generateOrnamentationSingle(trained_orn, line):
	#Generate ornamentation for a single line of a chord skeleton
	observed = ornamentationObs(line)

	transition = trained_orn[0]
	emission = trained_orn[1]
	initial = trained_orn[2]
	state_space = trained_orn[3]
	emission_space = trained_orn[4]

	print(emission_space)

	ornamentation = vb.viterbi(observed, state_space, emission_space, transition, emission, initial)

	return ornamentation

def trainOrnamentation(dataset):
	#Train ornamentation system for all lines

	trained_orn_a = trainOrnamentationSingle(dataset, 'a')
	trained_orn_t = trainOrnamentationSingle(dataset, 't')
	trained_orn_b = trainOrnamentationSingle(dataset, 'b')

	return (trained_orn_a, trained_orn_t, trained_orn_b)

def generateOrnamentation(generated, trained_orn):
	#Generate ornamentation for all parts
	trained_orn_a = trained_orn[0]
	trained_orn_t = trained_orn[1]
	trained_orn_b = trained_orn[2]

	line_a = getLine(generated, 'a')
	line_t = getLine(generated, 't')
	line_b = getLine(generated, 'b')

	orn_a = generateOrnamentationSingle(trained_orn_a, line_a)
	orn_t = generateOrnamentationSingle(trained_orn_t, line_t)
	orn_b = generateOrnamentationSingle(trained_orn_b, line_b)

	return orn_a, orn_t, orn_b


def test(choraleNo, trained):
	chorale = harmIO.extract(choraleNo)
	
	chorale = transpose(chorale, 'C')

	melody = getLine(chorale, 's')

	melody = melody[:40]

	harm = generate_harm(melody, trained)
	print(harm)

	chords = generate_chords(melody, harm, trained)


	final = format(chorale, 'C',melody, chords[1], chords[2], chords[3], chords[4])


	return final
	
if __name__ == '__main__':

	dataset, _ = harmIO.data('All', '+')
	dataset = hp.transposeData(dataset)
	trained = train(dataset)
	generated = test(263, trained)

	#print(generated)
	harmIO.printChorale(generated)
	# print('Now ornament \n \n \n \n \n \n \n \n')

	# trained_orn = trainOrnamentation(dataset1)
	# orn_a, orn_t, orn_b = generateOrnamentation(generated, trained_orn)
	# print(orn_a, '\n', orn_t, '\n', orn_b)
	# formatted = formatOrnamentation(generated, orn_a, orn_t, orn_b)
	# printChorale(formatted)
