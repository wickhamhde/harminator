############################
#   Final Project Readme   #
############################

This project requires the use of numpy. This can be installed at the command line by:
pip install numpy

This project should work in Python 2; however it has been built and tested in Python 3
so use of Python 3 is advisable.

To run the system to generate a single chorale, run chord_skeleton.py. You will be 
prompted for a tonality and a chorale number to choose from the dataset. It is
worth noting that the dataset numbering runs from 1 through to 389; the missing 
chorales are 121, 133, 199, 227,236,240, 297, 336, 350, and 382. Choosing these chorales
will cause the system to fail as the chorale will be extracted as null.

The output of the generated chorales is not yet in traditional musical notation;
however to be able to listen back I would recommend Musescore 2 as a musical notation
program for transcription and playback usage.