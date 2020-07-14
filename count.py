import os, os.path
import sys

arg = int(sys.argv[1])

location = ''

if arg == 0:
	location = '/front/straight'
elif arg == 1:
	location = '/front/correction'

DIR = './Conv_LSTM/test' + location

cpt = sum([len(files) for r, d, files in os.walk(DIR)])

print(cpt)