import operator
import os.path
import random
import harmIO
import math
import numpy as np
import harm_skeleton as hs
import chord_skeleton as cs
import ornamentation as orn
import helper as hp

def sameNote(n1, n2):
	if harmIO.raw(n1) == harmIO.raw(n2):
		return True
	else:
		return False
	

def sameChord(s1,a1,t1,b1,s2,a2,t2,b2):
	if sameNote(s1, s2) and sameNote(a1, a2) and sameNote(t1, t2) and sameNote(b1, b2):
		return True
	else:
		return False

#Small errors

def parallels_help(top1, top2, bottom1, bottom2, phrase):
	#Parallel fifths/octaves
	if interval(top1, bottom1) == 7 and interval(top2, bottom2) == 7 and sameNote(top1, top2) and sameNote(bottom1, bottom2):
		return -1
	elif interval(top1, bottom1) == 12 and interval(top2, bottom2) == 12 and sameNote(top1, top2) and sameNote(bottom1, bottom2):# and phrase != 1:
		return -1
	else:
		return 0

def parallels(chorale, phrase, n, verbose=False):
	para_sum = 0
	for i in range(5,8):
		temp = parallels_help(chorale[4][n], chorale[4][n+1], chorale[i][n], chorale[i][n+1], phrase[n+1])
		para_sum += temp

	for i in range(6,8):
		temp = parallels_help(chorale[5][n], chorale[5][n+1], chorale[i][n], chorale[i][n+1], phrase[n+1])
		para_sum += temp
	temp = parallels_help(chorale[6][n], chorale[6][n+1], chorale[7][n], chorale[7][n+1], phrase[n+1])
	para_sum += temp

	if para_sum != 0:
		if verbose == True:
			print('Parallels, line', str(n))
			print(chorale[2][n], chorale[3][n], chorale[4][n], chorale[5][n])
			print(chorale[2][n+1], chorale[3][n+1], chorale[4][n+1], chorale[5][n+1])
		return para_sum

	return 0


				
def cross(upper, lower):
	#Parts crossing each other
	if interval(upper, lower, positive=False) < 0:
		return -1
	else:
		return 0

def repeatedChord(s1, s2, a1, a2, t1, t2, b1, b2, beat):
	#Repeated chord from weak to strong beat
	#REWRITE BARS SO FIRST OF BAR = 2, 3RD IN 4/4 = 1, OTHERS = 0
	#REWRITE USING SAME()
	if sameNote(s1,s2) and sameNote(a1,a2) and sameNote(t1, t2) and sameNote(b1, b2):
		return -1
	else:
		return 0

def repeatedBass(b1, b2, beat):
	#Repeated bass not from weak to strong beat
	if sameNote(b1, b2) and beat < 1:
		return -1
	else:
		return 0

def octavesSaAt(s,a,t):
	#More than octave between S/A, A/T
	count = 0
	if interval(s,a) > 12:
		#print(interval(s,a), 1)
		count += 1
	if interval(a,t) > 12:
		#print(interval(a,t), 2)
		count += 1
	return -count

def sixFour(s,a,t,b, n):
	#Function for detecting erronious note-doubling in 6/4 chords
	i1 = interval(s[n],b[n])
	i2 = interval(a[n],b[n])
	i3 = interval(t[n],b[n])

	intervals = [i1%12, i2%12, i3%12]

	sixth = 0
	fourth = 0

	for i in intervals:
		if i == 5:
			fourth += 1
		elif i == 9:
			sixth += 1
		elif i != 0:
			return 0

	if fourth > 1:
		return -1

	else:
		return 0

def fiveThree(s,a,t,b, n):
	#Function for detecting erronious note-doubling in 5/3 chords
	i1 = interval(s[n],b[n])
	i2 = interval(a[n],b[n])
	i3 = interval(t[n],b[n])

	intervals = [i1%12, i2%12, i3%12]

	root = 0
	third = 0
	fifth = 0

	for i in intervals:
		if i == 4 or i == 3:
			third += 1
		elif i == 7:
			fifth += 1
		elif i != 0:
			return 0

	if third == 0:
		return -1

	else:
		return 0

def sixThree(s,a,t,b, n):
	#Function for detecting erronious note-doubling in 5/3 chords
	#Fix for corner cases of mismatching 6 and 3
	i1 = interval(s[n],b[n])
	i2 = interval(a[n],b[n])
	i3 = interval(t[n],b[n])
	i4 = 0

	intervals = [i1%12, i2%12, i3%12, 0]

	root = 0
	third = 0
	fifth = 0

	# i4 = 0
	# i3 = 0
	# i8 = 0
	# i9 = 0 

	for i in intervals:
		if i == 4 or i == 3:
			fifth += 1
			# if i == 3:
			# 	i3 += 1
			# else:
			# 	i4 += 1
		elif i == 8 or i == 9:
			root += 1
			# if i == 8:
			# 	i8 += 1
			# else:
			# 	i9 += 1
		elif i == 0:
			third += 1
		else:
			return 0

	# if i8 > 0 and i3 >0:
	# 	print('Mismatch 8 and 3')
	# 	print(s[n],a[n],t[n],b[n], n)
	# elif i9 > 0 and i4 >0:
	# 	print('Mismatch 9 and 4')

	if root == 0 or third == 0 or fifth == 0:
		return -1

	else:
		return 0

def doubledLeading(s,a,t,b,key, harmony):
	leading = 0
	for note in [s,a,t,b]:
		#FIX FOR MINOR KEYS
		if interval(note,key)%12 == 1:
			leading += 1
	#Allow doubled third in dominant chord
	if leading >= 2 and listed(harmony)[0] != 'D':
		return -1
	else:
		return 0

def outOfRange(s,a,t,b):
	if interval(s, 'G 2', positive=False) > 0 or interval(s, 'C 1', positive=False) < 0:
		return -1
	elif interval(a, 'E 2', positive=False) > 0 or interval(a, 'F 0', positive=False) < 0:
		return -1
	elif interval(t, 'G 1', positive=False) > 0 or interval(t, 'B -1', positive=False) < 0:
		return -1
	elif interval(b, 'E 1', positive=False) > 0 or interval(a, 'E -1', positive=False) < 0:
		return -1
	else:
		return 0

def dissonance_help(n1,n2):

	mild = [2, 10]
	sharp = [1, 11]
	if interval(n1,n2)%12 in mild:
		return -1
	elif interval(n1,n2)%12 in sharp:
		return -2
	else:
		return 0

def dissonance(chorale, n):
	diss_sum = 0
	for i in range(5,8):
		temp = dissonance_help(chorale[4][n], chorale[i][n])
		diss_sum += temp

	for i in range(6,8):
		temp = dissonance_help(chorale[5][n], chorale[i][n])
		diss_sum += temp
	temp = dissonance_help(chorale[6][n], chorale[7][n])
	diss_sum += temp

	if diss_sum > 1:
		return diss_sum

	return 0

#NOT ACTING ON ALL LIST ITEMS
def markable(chorale):
	new = [chorale[0],chorale[1], [],[],[],[],[],[],[]]
	done = False
	i = 0
	print(len(chorale[2]))
	while not done:
		if i>= len(chorale[2]):
			done = True
			print(i, 'donezo')
		else:
			# switch = True
			temp = []
			for j in range(2,9):
				temp.append(chorale[j][i])
				# if harmIO.isDashed(chorale[j][i]) == True:
					
				# 	switch = False
			if switch == True:
				
				for j in range(2,9):
					#print(i)
					new[j].append(chorale[j][i])
		i += 1		

	return new

def markA2(chorale):
	phrase, bar, soprano, alto, tenor, bass = chorale[2], chorale[3], chorale[4], chorale[5], chorale[6], chorale[7]

	scores = []
	for n in range(len(soprano)):
		score = 2
		errors = 0

		if n < len(soprano)-1:
			#Parallels
			errors += parallels(chorale, phrase, n)

			#Repeated chords
			temp_r = repeatedChord(soprano[n], soprano[n+1], alto[n], alto[n+1], tenor[n], tenor[n+1], bass[n], bass[n+1], bar[n])
			errors+=temp_r
			# if temp_r != 0:
			# 		print('Repeated chord, line', str(n))

		#Crossing lines
		for i in range(5,8):
			temp_c = cross(soprano[n], chorale[i][n])
			errors += temp_c
			# if temp_c != 0:
			# 	print('Crossed lines, line', str(n))

		for i in range(6,8):
			temp_c = cross(alto[n], chorale[i][n])
			errors += temp_c
			# if temp_c != 0:
			# 	print('Crossed lines, line', str(n))
		errors += cross(tenor[n], bass[n])

		#Interval greater than octave between S and A, A and T
		temp_oct = octavesSaAt(soprano[n], alto[n], tenor[n])
		errors += temp_oct
		# if temp_oct != 0:
		# 		print('Octaves error, line', str(n))
				#print(soprano[n], alto[n], tenor[n], bass[n])

		#Poor note doubling in six-four chord
		t64 = sixFour(soprano, alto, tenor, bass, n)
		if n > 0:
			if sixFour(soprano, alto, tenor, bass, n-1) != 0 and sameChord(soprano[n], alto[n], tenor[n], bass[n], soprano[n-1], alto[n-1], tenor[n-1], bass[n-1]) == False:
				# print('Six-Four error line', str(n))
				#print(soprano[n], alto[n], tenor[n], bass[n])
				errors += t64

		#Poor note doubling in six-three chord
		t63 = sixThree(soprano, alto, tenor, bass,n)
		if n > 0:
			if sixThree(soprano, alto, tenor, bass, n-1) != 0 and sameChord(soprano[n], alto[n], tenor[n], bass[n], soprano[n-1], alto[n-1], tenor[n-1], bass[n-1]) == False:
				# print('Six-Three error line', str(n))
				#print(soprano[n], alto[n], tenor[n], bass[n])
				errors += t63

		#Poor note doubling in five-three chord
		t53 = fiveThree(soprano, alto, tenor, bass,n)
		if n > 0:
			if fiveThree(soprano, alto, tenor, bass, n-1) != 0 and sameChord(soprano[n], alto[n], tenor[n], bass[n], soprano[n-1], alto[n-1], tenor[n-1], bass[n-1]) == False:
				# print('Five-Three error line', str(n))
				#print(soprano[n], alto[n], tenor[n], bass[n])
				errors += t53

		#Parts go out of vocal range
		temp_range = outOfRange(soprano[n], alto[n], tenor[n], bass[n])
		errors += temp_range
		# if temp_range != 0:
		# 		print('Range error, line', str(n))
				#print(soprano[n], alto[n], tenor[n], bass[n])

		temp_diss = dissonance(chorale, n)
		errors += temp_diss


		scores.append(score+errors)
	
	if (sum(scores)/( len(soprano)*2)) >0:
		return (sum(scores)/( len(soprano)*2))
	else:
		return 0


def scramble(chorale, iterations):
	temp_chorale = chorale
	for i in range(iterations):
		#Select 2 random notes, swap them
		line1 = random.randint(3,5)
		line2 = random.randint(3,5)
		beat1 = random.randint(0,len(chorale[2])-1)
		beat2 = random.randint(0,len(chorale[2])-1)


		#Store first in temp
		temp = temp_chorale[line1][beat1]
		#Swap second over to first
		temp_chorale[line1][beat1] = temp_chorale[line2][beat2]
		#Swap temp over to second
		temp_chorale[line2][beat2] = temp

	return temp_chorale

##############################################################################################################################################
#															HARMONIC SKELETON TESTING 														 #
##############################################################################################################################################

##################################################################################
#									Version 1									 #
##################################################################################

def harmCrossEntropySpacesV1(dataset):
	#Train the probabilities of melody sections producing harmony symbols

	state_space = []
	emission_space = []
	for chorale in dataset:
		line = hp.getLine(chorale, 's')
		melody_crotchets = hp.crotchets(line)
		harm_skel = hp.rawItems(hp.getLine(chorale, 'h'))

		T = len(harm_skel)
		for t in range(T):
			if melody_crotchets[t] not in emission_space:
				emission_space.append(melody_crotchets[t])
			if harm_skel[t] not in state_space:
				state_space.append(harm_skel[t])

	return state_space, emission_space

def trainHarmCrossEntropyV1(dataset, state_space, emission_space):
	#Train cross entropy analysis system

	probabilities = np.ones((len(emission_space), len(state_space)))

	for chorale in dataset:
			melody_crotchets = hp.crotchets(hp.getLine(chorale, 's'))
			harm_skel = hp.rawItems(hp.getLine(chorale, 'h'))
			T = len(melody_crotchets)
			for t in range(T):
				melodyInd = emission_space.index(melody_crotchets[t])
				harmInd = state_space.index(harm_skel[t])
				probabilities[melodyInd][harmInd] += 1

	probabilities = hp.normalise(probabilities)

	return probabilities

##################################################################################
#									Version 2									 #
##################################################################################

def harmCrossEntropySpacesV2(dataset):
	#Train the probabilities of melody sections producing harmony symbols

	state_space = []
	emission_space = []
	for chorale in dataset:
		line = hp.getLine(chorale, 's')
		melody_sections = hp.getCrotchetSections(line)
		harm_skel = hp.rawItems(hp.getLine(chorale, 'h'))

		T = len(harm_skel)
		for t in range(T):
			if melody_sections[t] not in emission_space:
				emission_space.append(melody_sections[t])
			if harm_skel[t] not in state_space:
				state_space.append(harm_skel[t])

	return state_space, emission_space


def trainHarmCrossEntropyV2(dataset, state_space, emission_space):
	#Train cross entropy analysis system

	probabilities = np.ones((len(emission_space), len(state_space)))

	for chorale in dataset:
			melody_sections = hp.getCrotchetSections(hp.getLine(chorale, 's'))
			harm_skel = hp.rawItems(hp.getLine(chorale, 'h'))
			T = len(harm_skel)
			for t in range(T):
				melodyInd = emission_space.index(melody_sections[t])
				harmInd = state_space.index(harm_skel[t])
				probabilities[melodyInd][harmInd] += 1

	probabilities = hp.normalise(probabilities)

	return probabilities

##################################################################################
#									Version 3									 #
##################################################################################

def harmCrossEntropySpacesV3(dataset):
	#Train the probabilities of melody sections producing harmony symbols

	state_space = []
	emission_space = []
	for chorale in dataset:
		line = hp.getLine(chorale, 's')
		beats = hs.getBeat(chorale)
		melody_sections = hp.getCrotchetSections(line)
		harm_skel = hp.rawItems(hp.getLine(chorale, 'h'))

		T = len(harm_skel)
		for t in range(T):
			item = (melody_sections[t], beats[t])
			if item not in emission_space:
				emission_space.append(item)
			if harm_skel[t] not in state_space:
				state_space.append(harm_skel[t])

	return state_space, emission_space

def trainHarmCrossEntropyV3(dataset, state_space, emission_space):
	#Train cross entropy analysis system

	probabilities = np.ones((len(emission_space), len(state_space)))

	for chorale in dataset:
			melody_sections = hp.getCrotchetSections(hp.getLine(chorale, 's'))
			beats = hs.getBeat(chorale)

			harm_skel = hp.rawItems(hp.getLine(chorale, 'h'))
			T = len(melody_sections)
			for t in range(T):
				item = (melody_sections[t], beats[t])
				itemInd = emission_space.index(item)
				harmInd = state_space.index(harm_skel[t])
				probabilities[itemInd][harmInd] += 1

	probabilities = hp.normalise(probabilities)

	return probabilities

##################################################################################
#									Version 4									 #
##################################################################################

def harmCrossEntropySpacesV4(dataset):
	#Train the probabilities of melody sections producing harmony symbols

	state_space = []
	emission_space = []
	for chorale in dataset:
		state_space = hs.getHarmStateSpaceV4(chorale, state_space)

		line = hp.getLine(chorale, 's')
		melody_sections = hp.getCrotchetSections(line)

		T = len(melody_sections)
		for t in range(T):
			if melody_sections[t] not in emission_space:
				emission_space.append(melody_sections[t])

	return state_space, emission_space

def trainHarmCrossEntropyV4(dataset, state_space, emission_space):
	#Train cross entropy analysis system

	probabilities = np.ones((len(emission_space), len(state_space)))

	for chorale in dataset:
			melody_sections = hp.getCrotchetSections(hp.getLine(chorale, 's'))
			harm_skel = hp.rawItems(hp.getLine(chorale, 'h'))
			T = len(melody_sections)
			for t in range(T):
				melodyInd = emission_space.index(melody_sections[t])
				harmInd = state_space.index(harm_skel[t])
				probabilities[melodyInd][harmInd] += 1

	probabilities = hp.normalise(probabilities)

	return probabilities

##################################################################################
#									Version 5									 #
##################################################################################

def harmCrossEntropySpacesV5(dataset):
	#Train the probabilities of melody sections producing harmony symbols
	state_space = []
	emission_space = []
	for chorale in dataset:
		state_space = hs.getHarmStateSpaceV4(chorale, state_space)

		line = hp.getLine(chorale, 's')
		melody_sections = hp.getCrotchetSections(line)
		beats = hs.getBeat(chorale)

		T = len(melody_sections)
		for t in range(T):
			item = (melody_sections[t], beats[t])
			if item not in emission_space:
				emission_space.append(item)

	return state_space, emission_space

def trainHarmCrossEntropyV5(dataset, state_space, emission_space):
	#Train cross entropy analysis system

	probabilities = np.ones((len(emission_space), len(state_space)))

	for chorale in dataset:
			melody_sections = hp.getCrotchetSections(hp.getLine(chorale, 's'))
			beats = hs.getBeat(chorale)
			harm_skel = hp.rawItems(hp.getLine(chorale, 'h'))

			T = len(melody_sections)
			for t in range(T):
				item = (melody_sections[t], beats[t])
				itemInd = emission_space.index(item)
				harmInd = state_space.index(harm_skel[t])
				probabilities[itemInd][harmInd] += 1

	probabilities = hp.normalise(probabilities)

	return probabilities



##############################################################################################################################################
#																CHORD SKELETON TESTING 														 #
##############################################################################################################################################

def chordCrossEntropySpaces(dataset):
	#State and emissio space for chord skeleton cross entropy analysis
	state_space = cs.chordStateSpaceV2(dataset)
	emission_space = cs.chordEmissionSpaceV2(dataset, 3)

	return state_space, emission_space

def trainChordCrossEntropy(dataset, state_space, emission_space):
	#Train cross entropy analysis system

	probabilities = np.ones((len(emission_space), len(state_space)))

	for chorale in dataset:
		observed = cs.chordObsV2(chorale, 1)
		state_seq = cs.chordStateSeqV2(chorale)
		T = len(observed)
		for t in range(T):
			obsInd = emission_space.index(observed[t])
			stateInd = state_space.index(state_seq[t])
			probabilities[obsInd][stateInd] += 1

	probabilities = hp.normalise(probabilities)

	return probabilities

##############################################################################################################################################
#																ORNAMENTATION TESTING 														 #
##############################################################################################################################################

def ornCrossEntropyObs(chorale):
	#Get observed sequence for CE analysis
	observed = []

	soprano_crotchets = hp.crotchets(hp.getLine(chorale, 's'))
	alto_crotchets = hp.crotchets(hp.getLine(chorale, 'a'))
	tenor_crotchets = hp.crotchets(hp.getLine(chorale, 't'))
	bass_crotchets = hp.crotchets(hp.getLine(chorale, 'b'))	
	
	for t in range(len(chorale)):
		observed.append((soprano_crotchets[t], alto_crotchets[t], tenor_crotchets[t], bass_crotchets[t]))

	return observed

def ornCrossEntropyStateSeq(chorale):
	#Get state space for CE analysis

	state_seq = []
	alto_sections = hp.getCrotchetSections(hp.getLine(chorale, 'a'))
	tenor_sections = hp.getCrotchetSections(hp.getLine(chorale, 't'))
	bass_sections = hp.getCrotchetSections(hp.getLine(chorale, 'b'))	

	for t in range(len(chorale)):
		state_seq.append((orn.semiquaverIntervals(alto_sections[t]), orn.semiquaverIntervals(tenor_sections[t]), orn.semiquaverIntervals(bass_sections[t])))

	return state_seq

def ornCrossEntropySpaces(dataset):
	#Get state and emission spaces for testing ornamentation
	#State space: ornamented segments in ATB
	#Emission space: initial crotchets of SATB
	state_space = []
	emission_space = []
	for chorale in dataset:
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

			if (orn.semiquaverIntervals(alto_sections[t]), orn.semiquaverIntervals(tenor_sections[t]), orn.semiquaverIntervals(bass_sections[t])) not in state_space:
				state_space.append((orn.semiquaverIntervals(alto_sections[t]), orn.semiquaverIntervals(tenor_sections[t]), orn.semiquaverIntervals(bass_sections[t])))

	return state_space, emission_space

def trainOrnCrossEntropy(dataset, state_space, emission_space):
	#Train cross entropy analysis for dataset

	probabilities = np.ones((len(emission_space), len(state_space)))

	for chorale in dataset:
		soprano = hp.getLine(chorale, 's')
		alto = hp.getLine(chorale, 'a')
		tenor = hp.getLine(chorale, 't')
		bass = hp.getLine(chorale, 'b')

		soprano_crotchets = hp.crotchets(soprano)
		alto_crotchets = hp.crotchets(alto)
		tenor_crotchets = hp.crotchets(tenor)
		bass_crotchets = hp.crotchets(bass)

		alto_sections = hp.getCrotchetSections(alto)
		tenor_sections = hp.getCrotchetSections(tenor)
		bass_sections = hp.getCrotchetSections(bass)

		T = len(alto_sections)

		for t in range(T):
			emInd = emission_space.index((soprano_crotchets[t], alto_crotchets[t], tenor_crotchets[t], bass_crotchets[t]))
			stateInd = state_space.index((orn.semiquaverIntervals(alto_sections[t]), orn.semiquaverIntervals(tenor_sections[t]), orn.semiquaverIntervals(bass_sections[t])))

			probabilities[emInd][stateInd] += 1

	probabilities = hp.normalise(probabilities)

	return probabilities

##############################################################################################################################################
#																	GENERAL FUNCTIONS														 #
##############################################################################################################################################


##################################################################################
#							Functions for running tests							 #
##################################################################################


def crossEntropy(observed, state_seq, state_space, emission_space, probabilities):
	#Find the cross entropy per symbol
	logSum = 0

	T = len(observed)

	for t in range(T):
		try:
			obsInd = emission_space.index(observed[t])
			stateInd = state_space.index(state_seq[t])
			logSum += math.log(probabilities[obsInd][stateInd], 2)
		except:
			print('Entry not found')

	return -logSum/T


def getTestChorales(dataset):
	#Randomly select a set of chorales to be used as tests
	
	testChorales = []


	for i in range(5):
		done = False
		while not done:
			test = random.randint(0, len(dataset)-1)
			if test not in testChorales:
				testChorales.append(test)
				done=True

	return testChorales

def extractTestChorales(testChorales, dataset):
	#Return fully extracted test data

	chorales = []

	for test in testChorales:
		chorale = dataset[test]
		chorales.append(chorale)

	return chorales

def test_harm_skeletons(tonality):
	#Function to test and compare harmonic skeleton models

	dataset, _ = harmIO.data('All', tonality)
	dataset = hp.transposeData(dataset)

	testChorales = getTestChorales(dataset)
	print(testChorales)
	extracted = extractTestChorales(testChorales, dataset)

	generated_harm_skels_v1 = []
	generated_harm_skels_v2 = []
	generated_harm_skels_v3 = []
	generated_harm_skels_v4 = []
	generated_harm_skels_v5 = []
	real_harm_skels = []

	#train for generating test harmonic skeletons
	trainedGen1 = hs.train_harm_skel(dataset, 1)
	trainedGen2 = hs.train_harm_skel(dataset, 2)
	trainedGen3 = hs.train_harm_skel(dataset, 3)
	trainedGen4 = hs.train_harm_skel_order_2(dataset, 4)
	trainedGen5 = hs.train_harm_skel_order_2(dataset, 5)


	#Generate harmonies to be tested
	for chorale in extracted:
		melody = hp.getLine(chorale, 's')
		harmony = hp.getLine(chorale, 'h')
		harm_raw = hp.rawItems(harmony)
		beats = hs.getBeat(chorale)

		#Generate alternate harmonic skeletons
		print('Generating harmonies', extracted.index(chorale))
		harm_skel_1 = hs.generate_harm(melody, trainedGen1, 1)
		harm_skel_2 = hs.generate_harm(melody, trainedGen2, 2)
		harm_skel_3 = hs.generate_harm(melody, trainedGen3, 3, beats)
		harm_skel_4 = hs.generate_harm_order_2(melody, trainedGen4, 4)
		harm_skel_5 = hs.generate_harm_order_2(melody, trainedGen4, 5, beats)


		#Store
		generated_harm_skels_v1.append(harm_skel_1)
		generated_harm_skels_v2.append(harm_skel_2)
		generated_harm_skels_v3.append(harm_skel_3)
		generated_harm_skels_v4.append(harm_skel_4)
		generated_harm_skels_v5.append(harm_skel_5)
		real_harm_skels.append(harm_raw)

	#State and emission spaces for differet versions' cross entropy testing
	state_space1, emission_space1 = harmCrossEntropySpacesV1(dataset)
	state_space2, emission_space2 = harmCrossEntropySpacesV2(dataset)
	state_space3, emission_space3 = harmCrossEntropySpacesV3(dataset)
	state_space4, emission_space4 = harmCrossEntropySpacesV4(dataset)
	state_space5, emission_space5 = harmCrossEntropySpacesV5(dataset)


	#Train for cross entropy testing	
	probs1 = trainHarmCrossEntropyV1(dataset, state_space1, emission_space1)
	probs2 = trainHarmCrossEntropyV2(dataset, state_space2, emission_space2)
	probs3 = trainHarmCrossEntropyV3(dataset, state_space3, emission_space3)
	probs4 = trainHarmCrossEntropyV4(dataset, state_space4, emission_space4)
	probs5 = trainHarmCrossEntropyV5(dataset, state_space5, emission_space5)


	#Lists to store cross entropy scores
	scores1_gen = []
	scores1_real = []
	scores2_gen = []
	scores2_real = []
	scores3_gen = []
	scores3_real = []
	scores4_gen = []
	scores4_real = []
	scores5_gen = []
	scores5_real = []

	#Test skeletons
	for i in range(5):
		chorale = extracted[i]
		melody = hp.getLine(chorale, 's')
		beats = hs.getBeat(chorale)
		harmony = hp.getLine(chorale, 'h')

		observed1 = hp.crotchets(melody)
		observed2 = hs.getHarmObsV2(melody)
		observed3 = hs.getHarmObsV3(melody, beats)
		observed4 = hs.getHarmObsV2(melody)
		observed5 = hs.getHarmObsV3(melody, beats)


		harm_skel_1 = generated_harm_skels_v1[i]
		harm_skel_2 = generated_harm_skels_v2[i]
		harm_skel_3 = generated_harm_skels_v3[i]
		harm_skel_4 = generated_harm_skels_v4[i]
		harm_skel_5 = generated_harm_skels_v5[i]

		harm_skel_real = hp.rawItems(harmony)

		scores1_gen.append(crossEntropy(observed1, harm_skel_1, state_space1, emission_space1, probs1))
		scores1_real.append(crossEntropy(observed1, harm_skel_real, state_space1, emission_space1, probs1))
		scores2_gen.append(crossEntropy(observed2, harm_skel_2, state_space2, emission_space2, probs2))
		scores2_real.append(crossEntropy(observed2, harm_skel_real, state_space2, emission_space2, probs2))
		scores3_gen.append(crossEntropy(observed3, harm_skel_3, state_space3, emission_space3, probs3))
		scores3_real.append(crossEntropy(observed3, harm_skel_real, state_space3, emission_space3, probs3))
		scores4_gen.append(crossEntropy(observed4, harm_skel_4, state_space4, emission_space4, probs4))
		scores4_real.append(crossEntropy(observed4, harm_skel_real, state_space4, emission_space4, probs4))
		scores5_gen.append(crossEntropy(observed5, harm_skel_5, state_space5, emission_space5, probs5))
		scores5_real.append(crossEntropy(observed5, harm_skel_real, state_space5, emission_space5, probs5))


	mean_gen_1 = sum(scores1_gen)/5	
	mean_real_1 = sum(scores1_real)/5
	mean_gen_2 = sum(scores2_gen)/5	
	mean_real_2 = sum(scores2_real)/5
	mean_gen_3 = sum(scores3_gen)/5	
	mean_real_3 = sum(scores3_real)/5
	mean_gen_4 = sum(scores4_gen)/5	
	mean_real_4 = sum(scores4_real)/5
	mean_gen_5 = sum(scores5_gen)/5	
	mean_real_5 = sum(scores5_real)/5

	print(mean_gen_1, mean_real_1, mean_gen_2, mean_real_2, mean_gen_3, mean_real_3, mean_gen_4, mean_real_4,mean_gen_5, mean_real_5)

def test_chord_skeletons(tonality):
	#Function to test and compare harmonic skeleton models

	dataset, _ = harmIO.data('All', tonality)
	dataset = hp.transposeData(dataset)

	testChorales = getTestChorales(dataset)
	print(testChorales)
	extracted = extractTestChorales(testChorales, dataset)

	generated_chord_skels = []
	real_chord_skels = []

	#Train for harmonic skeleton generation for generating test chord skeletons
	trained_harm = hs.train_harm_skel(dataset, 3)
	#Train chord skeleton for generating test chord skeletons
	trained_chords = cs.train_chord_skelV2(dataset, 3)
		
	#Generate chords to be tested
	for chorale in extracted:
		
		melody = hp.getLine(chorale, 's')
		beats = hs.getBeat(chorale)
		#Generate harmonc skeleton
		print('Generating harmonies', extracted.index(chorale))
		harm_skel = hs.generate_harm(melody, trained_harm, 3, beats)

		print('Generating chord skeleton')
		chord_skel = cs.generate_chordsV2(melody, harm_skel, trained_chords, 1)


		#Store
		generated_chord_skels.append(chord_skel)
		
		real_chord_skel = cs.chordStateSeqV2(chorale)
		
		real_chord_skels.append(real_chord_skel)

	#State and emission spaces for differet versions' cross entropy testing
	state_space, emission_space = chordCrossEntropySpaces(dataset)


	#Train for cross entropy testing	
	probs = trainChordCrossEntropy(dataset, state_space, emission_space)

	#Lists to store cross entropy scores
	scores_gen = []
	scores_real = []

	#Test skeletons
	for i in range(5):
		chorale = extracted[i]
		melody = hp.getLine(chorale, 's')
		harmony = hp.getLine(chorale, 'h')

		observed = cs.chordObsV2(chorale, 1)

		chord_skel = generated_chord_skels[i]

		chord_skel_real = cs.chordStateSeqV2(chorale)

		scores_gen.append(crossEntropy(observed, chord_skel, state_space, emission_space, probs))
		scores_real.append(crossEntropy(observed, chord_skel_real, state_space, emission_space, probs))


	mean_gen = sum(scores_gen)/5	
	mean_real = sum(scores_real)/5


	print(mean_gen, mean_real)

def test_ornamentation(tonality):
	#Function to test and compare harmonic skeleton models

	dataset, _ = harmIO.data('All', tonality)
	dataset = hp.transposeData(dataset)

	testChorales = getTestChorales(dataset)
	print(testChorales)
	extracted = extractTestChorales(testChorales, dataset)

	generated_orns1 = []
	generated_orns2 = []
	generated_orns3 = []
	generated_orns4 = []

	#Train for harmonic skeleton generation for generating test chord skeletons
	trained_harm = hs.train_harm_skel(dataset, 3)
	#Train chord skeleton for generating test chord skeletons
	trained_chords = cs.train_chord_skelV2(dataset, 3)

	trained_orn1_fwd = orn.trainOrnamentationV1(dataset, False)
	trained_orn1_bwd = orn.trainOrnamentationV1(dataset, True)

	trained_orn2_semis = orn.trainOrnamentation2(dataset, True)
	trained_orn2_nosemis = orn.trainOrnamentation2(dataset, False)
		
	#Generate chords to be tested
	for chorale in extracted:
		
		chorale = list(chorale)
		melody = hp.getLine(chorale, 's')
		beats = hs.getBeat(chorale)
		#Generate harmonc skeleton
		print('Generating harmonies')
		harm_skel = hs.generate_harm(melody, trained_harm, 3, beats)

		print('Generating chord skeleton')
		chord_skel = cs.generate_chordsV2(melody, harm_skel, trained_chords, 1)

		formatted_chord_skel = cs.formatChordSkel(melody, chord_skel, harm_skel, chorale)
		
		#HMM
		orn1 = orn.generate_ornamentationV1(formatted_chord_skel, trained_orn1_bwd, True)
		orn2 = orn.generate_ornamentationV1(formatted_chord_skel, trained_orn1_fwd, False)
		#Context 2 Markov Model
		orn3 = orn.generate_ornamentation2(formatted_chord_skel, trained_orn2_nosemis, semiquavers=False)
		orn4 = orn.generate_ornamentation2(formatted_chord_skel, trained_orn2_semis, semiquavers=True)

		orn1_f = orn.formatOrnamentation(chorale, orn1[0], orn1[1], orn1[2])
		orn2_f = orn.formatOrnamentation(chorale, orn2[0], orn2[1], orn2[2])
		orn3_f = orn.formatOrnamentation(chorale, orn3[0], orn3[1], orn3[2])
		orn4_f = orn.formatOrnamentation(chorale, orn4[0], orn4[1], orn4[2])

		generated_orns1.append(orn1_f)
		generated_orns2.append(orn2_f)
		generated_orns3.append(orn3_f)
		generated_orns4.append(orn4_f)

		try:
			print('Verison 1')
			harmIO.printChorale(orn1_f)
			print('Verison 2')
			harmIO.printChorale(orn2_f)
			print('Verison 3')
			harmIO.printChorale(orn3_f)
			print('Verison 4')
			harmIO.printChorale(orn4_f)
		except:
			print('Error')
		
	#State and emission spaces for differet versions' cross entropy testing
	state_space, emission_space = ornCrossEntropySpaces(dataset)

	#Train for cross entropy testing	
	probs = trainOrnCrossEntropy(dataset, state_space, emission_space)

	#Lists to store cross entropy scores
	scores_gen1 = []
	scores_gen2 = []
	scores_gen3 = []
	scores_gen4 = []
	scores_real = []

	#Test skeletons
	for i in range(5):
		chorale = extracted[i]

		observed = ornCrossEntropyObs(chorale)

		orn1 = ornCrossEntropyStateSeq(generated_orns1[i])
		orn2 = ornCrossEntropyStateSeq(generated_orns2[i])
		orn3 = ornCrossEntropyStateSeq(generated_orns3[i])
		orn4 = ornCrossEntropyStateSeq(generated_orns4[i]) 
		orn_real = ornCrossEntropyStateSeq(chorale)

		scores_gen1.append(crossEntropy(observed, orn1, state_space, emission_space, probs))
		scores_gen2.append(crossEntropy(observed, orn2, state_space, emission_space, probs))
		scores_gen3.append(crossEntropy(observed, orn3, state_space, emission_space, probs))
		scores_gen4.append(crossEntropy(observed, orn4, state_space, emission_space, probs))
		scores_real.append(crossEntropy(observed, orn_real, state_space, emission_space, probs))


	mean_gen1 = sum(scores_gen1)/5	
	mean_gen2 = sum(scores_gen2)/5	
	mean_gen3 = sum(scores_gen3)/5	
	mean_gen4 = sum(scores_gen4)/5	
	mean_real = sum(scores_real)/5


	print(mean_gen1,mean_gen2,mean_gen3,mean_gen4, mean_real)



if __name__ == '__main__':

	testType = input('Choose type of skeleton to test. h for harmonic, c for chords, or o for ornamentation.')
	tonality = input('Choose the tonality to test: ')
	if testType == 'h':
		test_harm_skeletons(tonality)
	elif testType == 'c':
		test_chord_skeletons(tonality)
	elif testType == '0':
		test_ornamentation(tonality)
	else:
		print('Please use valid input.')


	#HARMONY
	#Run 1 + chorales:  [53, 105, 170, 163, 83]
	#Run 1 + : 2.2948322925659133 2.9640494252633376 2.406966287832852 2.9338055914662724 2.60687359498327 3.0700019171223483 6.194881692371055 4.300609885245355 6.796835509354042 5.40078523625836
	#Run 1 - chorales: [165, 32, 3, 0, 129]
	#Run 1 - : 2.719680953816194 3.09751873993149 2.8091179772929697 3.012978669453647 2.9382809616011647 3.092735330253547 6.965589603938959 4.550441030015314 7.610718285042028 5.652042449826794

	#CHORDS
	#Run 1 + chorales: [35, 90, 33,22,66]
	#Run 1 - : 9.205 5.856
	#Run 2 + chorales: [168,92,174,108]
	#Run 1 - : 9.444 5.707

	#ORNAMENTATION
	#Run 1 - chorales: [156, 30, 159, 120, 161]
	#Run 1 - : 5.890223156163768 5.890223156163768 5.890223156163768 5.890223156163768 5.834383222617928
	#Run 1 + chorales: [75, 167, 30, 123, 110]
	#Run 1 + : 5.242999476886949 5.242999476886949 5.242999476886949 5.242999476886949 5.649706908851146