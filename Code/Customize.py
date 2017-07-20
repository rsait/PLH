'''
Created on Jan 13, 2016

@author: 
'''

import subprocess
import sys
import os

# In the command line you have to set the PLH_INSTALLATION variable with the directory where it has been cloned from Github.
# For example:
# export PLH_INSTALLATION=/home/PLH

plh_installation_directory = os.environ['PLH_INSTALLATION']

code_directory = plh_installation_directory + '/Code/' # this code has to be customized to your needs

sys.path.append(code_directory + '/Utils/')

# this code has to be customized to your needs

data_directory = plh_installation_directory + '/Data/' 
tmp_directory = plh_installation_directory + '/tmp/'

Ds_filename = data_directory + "CorpusTest.txt"
Dv_filename = data_directory + "Puntus.txt"

Ds_already_lemmatized = True
Dv_already_lemmatized = True
lemmatizedDs_filename = data_directory + "CorpusTestLemmatized.txt"
lemmatizedDv_filename = data_directory + "PuntusLemmatized.txt"

# this code has to be customized to your needs

M = (',','.','?','!','"',"'",'\/','\\') 
NV = 4 
RP = (0, 0, 0, 0) 

# this code has to be customized to your needs

def extract_verses (d):
    return d

with open(data_directory + 'auxVerbs.txt') as f:
    auxWords = f.read().splitlines()

'''
def lemmatize (w):
    if w in auxWords:
        return 'aux'
    return subprocess.check_output(["perl", code_directory + "lemmatizeWord.pl",w])
'''

def is_rhyme (w1, w2):
    rhyme = subprocess.check_output(['perl', code_directory + 'isRhyme.pl', w1, w2])
    if rhyme == 'True':
        return True
    if rhyme == 'False':
        return False
    return False
    
# default values for other parameters, which could also be customized

number_topics = 100
filtered_words = []
no_below = 5
no_above = 0.2

# customized values

filtered_words = ['dut', 'ni', 'zu', 'da', 'du', 'dute', 'zen', 'ere', 'gu', 'dugu', 'ez', 'bat', 'hori', 'hor', 'dira', 
            'baina', 'bi', 'zi', 'zut', 'zituzten', 'atzo', 'beste', 'dela']


