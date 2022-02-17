import numpy as np
import os.path
import viterbi_bach as vb
import math
import evaluateChorales as ec
import harmIO


def getKey(line):

	#Find out whether key is major or minor
	if list(line)[-2] == 'l':
		tonality = '-'
	else:
		tonality = '+'

	#Find the key centre
	# temp = list(line)[-7:]
	# if temp[0] == '=':
	# 	temp = temp[1:]
	# if temp[0] == ' ':
	# 	temp = temp[1:]

	# if temp[1] == '-':
	# 	return temp[0], tonality
	# else:
	# 	return (temp[0] + temp[1]), tonality

	if tonality == '+':
		temp = list(line)[:-5]
		if temp[-2] == ' ':
			key = temp[-1]
			return key, tonality
		else:
			key = temp[-2] + temp[-1]

			return key, tonality
	else:
		temp = list(line)[:-6]
		if temp[-2] == ' ':
			key = temp[-1]
			return key, tonality
		else:
			key = temp[-2] + temp[-1]
			return key, tonality

def isDashed(string):
	if not isinstance(string, str):
		return False
	elif string[-1] == '-':
		return True
	else:
		return False

#Append '-' to continued notes to differentiate from semiquavers
def continued(line):
	last = line[-1]
	if isDashed(last) == True:
		return last
	else:
		return last + '-'

def raw(item):
	#Version for strings (symbols, notes etc.)
	if isinstance(item, str):
		if isDashed(item):
			return item[:-1]
		else:
			return item
	else:
		#Version for tuples for sop+harmony 
		if len(item) == 2:
			new = []
			for i in item:
				if isDashed(i):
					new.append(i[:-1])
				else:
					new.append(i)
			return tuple(new)
		#Version for voicing structures (tuple length 3 or 4)
		elif len(item) >= 3:
			if item[-1] == '-':
				new = item[:-1]
				return tuple(new)
			else:
				return item

def dashed(string):
	if isDashed(string) == True:
		return string
	else:
		return string+'-'

def rawItems(symbols):
	rawList = []
	for symbol in symbols:
		if checkNew(symbol) == True:
			rawList.append(symbol)
	return rawList

def checkNew(string):
	if isDashed(string) == True:
		return False
	else:
		return True

def allDashed(args):
	#Returns False if all items dashed, True otherwise
	for item in args:
		if not isDashed(item):
			return False

	return True	

def notAllDashed(args):
	#Returns False if all items dashed, True otherwise
	for item in args:
		if not isDashed(item):
			return True

	return False

def isCrotchet(note):
	if isDashed(note[0]) and not notAllDashed(note[1:]):
		return True
	else:
		return False
def interval(n1, n2, positive=True):
	#Express interval between two notes as number of semitones

	notes = {'C ':0, 'H#':0, 'C#':1, 'Db':1, 'D ':2, 'D#':3, 'Eb':3, 'E ':4, 'E#':5, 'F ':5, 'F#':6, 'Gb':6, 'G ':7, 'G#':8, 'Ab':8, 'A ':9, 'A#':10, 'B ':10, 'H ':11}


	n1_l = list(raw(n1))
	n2_l = list(raw(n2))


	temp1 = n1_l[0] + n1_l[1]
	temp2 = n2_l[0] + n2_l[1]

	note1_val =0
	note2_val = 0

	if len(n1_l) == 3:
		note1_val = notes[temp1] + int(n1_l[2])*12
	elif len(n1_l) == 4:
		note1_val = notes[temp1] - 12

	if len(n2_l) == 3:
		note2_val = notes[temp2] + int(n2_l[2])*12

	elif len(n2_l) == 4:
		note2_val = notes[temp2] - 12



	#Replace with plain mod function
	diff = note1_val-note2_val

	if positive == True:
		if diff < 0:
			diff *= -1

	return diff

def crotchets(line, acceptMinims=False):
	#Return only the notes that fall on the crotchet beat
	new = []
	T = len(line)

	for t in range(T):
		if t%4 == 0:
			#Must account for minims
			if acceptMinims==False:
				new.append(raw(line[t]))
			else:
				new.append(line[t])
	return new

def transpose(chorale, target):
	currentKey = chorale[1]
	keys = {'C':0, 'H#':0, 'C#':1, 'Db':1, 'D':2, 'D#':3, 'Eb':3, 'E':4, 'E#':5, 'F':5, 'F#':6, 'Gb':6, 'G':7, 'G#':8, 'Ab':8, 'A':9, 'A#':10, 'B':10, 'H':11}


	notes = allNotes()

	#Find interval of transposition
	interval = keys[target]-keys[currentKey]
	#Minimise interval
	if interval == 0:
		return chorale
	elif interval >6:
		interval -= 12
	elif interval <-6:
		interval += 12

	transposed = [chorale[0], target, chorale[2], chorale[3], [],[],[],[],chorale[8]]
	#Go through each line
	for line in range(4,8):
		for note in chorale[line]:
			if isDashed(note) == False:
				transposed[line].append(transpose_single(interval, note, notes))
			else:
				
				transposed[line].append(continued(transposed[line]))
	return transposed


def transposeData(dataset, key='C'):
	#Transpose to key of C
	transposed = []
	for chorale in dataset:
		transposed.append(transpose(chorale, key))

	return transposed

def getCrotchetSections(line):
	#Get all crotchet-length sections of the line, returned as a list of length 4 tuples

	T = int(len(line)/4)

	sections = []

	for t in range(T):
		time = t*4

		section = tuple(line[time:time+4])

		sections.append(section)

	return sections

def sectionIntervals(section1, section2):
	newSection = []
	for i in range(4):
		newSection.append(ec.interval(section1[i], section2[i]))

	return newSection


def transpose_single(interval, note, notes):

	#Check that we actually need to tranpose
	if interval == 0:
		return note
	
	#Check for enharmonic versions of notes
	if note not in notes:
		note = enharmonic(note)

	note_index = notes.index(note)

	new_index = note_index + interval

	newNote = notes[new_index]

	return newNote

def allNotes():
	notes = []
	notes_raw = ['C ', 'C#', 'D ', 'Eb', 'E ', 'F ', 'F#', 'G ', 'Ab', 'A ', 'B ', 'H ']

	for i in range(-3, 5):
		for note in notes_raw:
			notes.append(note + str(i))

	return notes

def enharmonic(note):
	if not isinstance(note, str):
		return note
	octave = raw(note)[2:]
	note_start = note[:2]
	pairs = [('C ', 'H#'), ('C#', 'Db'), ('D#', 'Eb'), ('E#', 'F '), ('F#', 'Gb'), ('G#', 'Ab'), ('A#', 'B ')]

	for pair in pairs:
		if note_start in pair:
			index = pair.index(note_start)
			if index == 0:
				return pair[1]+octave
			else:
				return pair[0]+octave
	return note


if __name__ == '__main__':

	dataset, _ = harmIO.data(200, '-')
	

def getLine(chorale, line):
	if line == 's':
		return chorale[4]
	elif line == 'a':
		return chorale[5]
	elif line == 't':
		return chorale[6]
	elif line == 'b':
		return chorale[7]
	elif line == 'h':
		return chorale[8]
	else:
		print('Enter a valid line to return.')
        	
#FINE
def normalise(normal):

	if type(normal) is list:
		
		total = sum(normal)
		#Normalise so all probabilities add to 1
		if sum(normal) != 0:
			for i in range(len(normal)):
				normal[i] /= total
		else:
			print('No probabilities stored')


		return normal 

	else:
		#Normalise so that all rows add to 1

		for i in range(len(normal)):
			temp = sum(normal[i])
			for j in range(len(normal[i])):
				if sum(normal[i]) != 0:
					normal[i][j] /= temp
		return normal

def getItemsSingle(items, line):
	for n in range(len(line)):
		currentItem = raw(line[n])
		
		if currentItem not in items:
			items.append(currentItem)

	return items


def getSymbols(chorale, symbols):
	#Return all harmonic symbols used in a given chorale
	harmony = getLine(chorale, 'h')
	#print(harmony)
	symbols = getItemsSingle(symbols, harmony)

	return symbols


def getSingles(chorale, line, notes):
	#Return all different notes used in chorale
	#line = 's', 'a', 't' or 'b'
	lines = {'s':4, 'a':5, 't':6, 'b':7, 'h':8}

	voice = lines[line]

	notes = getItemsSingle(notes, chorale[voice])

	return notes

def fill_skel(line, melody):
	new = []

	for item in line:
		new.append(item)

		for _ in range(3):
			new.append(hp.dashed(new[-1]))

	remaining = len(melody) - len(new)

	for _ in range(remaining):
		new.append(hp.dashed(new[-1]))

	return new


#Function for extracting a particular set of transiiton matrices from train() output
def getTransition(trained, part):
	if part == 'h':
		return trained[0][0][0]
	elif part == 'a':
		return trained[0][1][0]
	elif part == 't':
		return trained[0][2][0]
	elif part == 'b':
		return trained[0][3][0]
	else:
		print('Choose valid part.')

#Function for extracting a particular emission matrix from train() output
def getEmission(trained, part):
	if part == 'h':
		return trained[0][0][1]
	elif part == 'a':
		return trained[0][1][1]
	elif part == 't':
		return trained[0][2][1]
	elif part == 'b':
		return trained[0][3][1]
	else:
		print('Choose valid part.')


#Function for extracting a particular initial probabilities from train() output
def getInitial(trained, part):
	if part == 'h':
		return trained[0][0][2]
	elif part == 'a':
		return trained[0][1][2]
	elif part == 't':
		return trained[0][2][2]
	elif part == 'b':
		return trained[0][3][2]
	else:
		print('Choose valid part.')