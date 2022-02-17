import os.path
import random
import numpy as np
import matplotlib.pyplot as plt
import helper as hp

def getPath(choraleNo):
	#Find file path for chorale
	if choraleNo < 10:
		path = r"music\bch00" + str(choraleNo) + ".txt"
	elif choraleNo <100 and choraleNo >= 10:
		path = r"music\bch0" + str(choraleNo) + ".txt"
	else:
		path = r"music\bch" + str(choraleNo) + ".txt"

	return path


def extract(choraleNo):
	#Turn text data into usable data
	path = getPath(choraleNo)




	tonality = ''

	phrase = []
	bar = []
	soprano = []
	alto = []
	tenor = []
	bass = []
	harmony = []

	file = open(path, 'r')

	count = 0
	for line in file:
		count += 1

		if count == 3:
			key, tonality = hp.getKey(line)	

		if count > 9:
			splitLine = line.split("\t")
			for i in range(len(splitLine)):
				splitLine[i] = splitLine[i].rstrip()


			#No change at all since last time step
			if splitLine[0] == '' and len(splitLine) == 1:
				phrase.append(hp.continued(phrase))
				bar.append(hp.continued(bar))
				soprano.append(hp.continued(soprano))
				alto.append(hp.continued(alto))
				tenor.append(hp.continued(tenor))
				bass.append(hp.continued(bass))
				harmony.append(hp.continued(harmony))

			else:
				#No change to Phrase
				if splitLine[0] == '':
					phrase.append(str('-'))
				#New phrase
				else:
					phrase.append('1')

				#No change to Bar
				if splitLine[1] == '':
					#Case for anacrusis
					if len(bar) == 0:
						bar.append('0-')
					else:
						bar.append(hp.continued(bar))
				#New bar
				else:
					bar.append(splitLine[1])

				#No change to Soprano
				if splitLine[2] == '' or splitLine[2] == 'P':
					soprano.append(hp.continued(soprano))
				else:
					soprano.append(splitLine[2])

				if splitLine[3] == '' or splitLine[3] == 'P':
					alto.append(hp.continued(alto))
				else:
					alto.append(splitLine[3])

				if splitLine[4] == '' or splitLine[4] == 'P':
					tenor.append(hp.continued(tenor))
				else:
					tenor.append(splitLine[4])

				if splitLine[5] == '' or splitLine[5] == 'P':
					bass.append(hp.continued(bass))
				else:
					bass.append(splitLine[5])

				if splitLine[6] == '':
					harmony.append(hp.continued(harmony))
				else:
					harmony.append(splitLine[6])

	chorale =  (tonality, key,phrase, bar, soprano, alto, tenor, bass, harmony)
	return chorale

def data(size, tonality, start=0, verbose=False):
	dataset = []
	done = False
	count = start
	count2 = 0

	if size == 'All':
		size = 389
	while count < 389:
		count += 1
		
		missing = [121, 133, 199, 227,236,240, 297, 336, 350, 382]

		if count not in missing:
		
			chorale = extract(count)
			
			if chorale != None:
				if chorale[0] == tonality:
					count2 += 1
					if verbose==True:
						print(count2, count)
					dataset.append(chorale)
		if len(dataset) == size or count == 389:
			done = True

	print(len(dataset), 'chorales in dataset')
	return dataset, count+1
	
def format(chorale, key,melody, alto, tenor, bass, harm_skel):
	newChorale = [key,chorale[1], chorale[2], chorale[3], melody, alto, tenor, bass, harm_skel]
	return newChorale

def printChorale(chorale):
	sop =hp.getLine(chorale, 's')
	alto =hp.getLine(chorale, 'a')
	ten =hp.getLine(chorale, 't')
	bass =hp.getLine(chorale, 'b')
	harm =hp.getLine(chorale, 'h')

	for n in range(len(sop)):
		temp_s = ''
		temp_a = ''
		temp_t = ''
		temp_b = ''
		temp_h = ''

		if hp.isDashed(sop[n]) == True:
			temp_s = '   '
		else:
			temp_s = sop[n]


		if hp.isDashed(alto[n]) == True:
			temp_a = '   '
		else:
			temp_a = alto[n]


		if hp.isDashed(ten[n]) == True:
			temp_t = '   '
		else:
			temp_t = ten[n]


		if hp.isDashed(bass[n]) == True:
			temp_b = '   '
		else:
			temp_b = bass[n]


		if hp.isDashed(harm[n]) == True:
			temp_h = '   '
		else:
			temp_h = harm[n]

		print(temp_s, '\t', temp_a, '\t',temp_t,'\t', temp_b,'\t', temp_h)

def formatOrnamentation(chorale, orn_a, orn_t, orn_b):
	#Format ornamentation into full chorale
	alto = []
	tenor = []
	bass = []

	old_a = chorale[5]
	old_t = chorale[6]
	old_b = chorale[7]

	for item in orn_a:
		for i in item:
			alto.append(i)
	for item in orn_t:
		for i in item:
			tenor.append(i)
	for item in orn_b:
		for i in item:
			bass.append(i)

	newChorale = [chorale[0],chorale[1], chorale[2], chorale[3], chorale[4], alto, tenor, bass, chorale[8]]

	return newChorale

if __name__ == '__main__':

	dataset_major = data('All', '+')
	dataset_minor = data('All', '-')
	            

