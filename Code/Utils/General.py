# -*- coding: utf-8 -*-
'''
Created on Apr 7, 2015

@author: José María Martínez-Otzeta

Collection of Python functions used in the Bertsobot project

'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import collections

def unlist(l):
    '''
    Description: Making a flat list out of list of lists.\n
    Input: List made up of sublists.\n
    Output: List with atomic elements.\n
    Example:\n
    C{In: unlist([[1,2], [7,4], [9,0]])}\n
    C{Out: [1, 2, 7, 4, 9, 0]}\n
    Related functions:\n
    @param l: list of lists
    @type l: C{list}
    @return: 
    @rtype: C{list}
    '''
    return [item for sublist in l for item in sublist]

def readLineNumberFromFile(number, filename):
    '''
    Description: Returns the line at the given position. The line count starts in zero. If there is no line with such number, the function returns None.\n
    Input: Number of line wanted and name of the file to read from.\n
    Output: String corresponding to the desired line.\n 
    Example:\n
    C{$cat a.txt}\n
    C{$First sentence}\n
    C{$Second sentence}\n
    C{$Third sentence}\n
    C{In: readLineNumberFromFile(2, 'a.txt')}\n
    C{Out: 'Third sentence'}\n
    C{In: result = readLineNumberFromFile(20, 'a.txt')}\n
    C{In: type(result)}\n
    C{Out: NoneType}\n
    C{In: result = readLineNumberFromFile(-100, 'a.txt')}\n
    C{In: type(result)}\n
    C{Out: NoneType}\n
    Related functions:\n
    @param number: line number
    @type number: C{int}
    @param filename: file name
    @type filename: C{str}
    @return: 
    @rtype: C{str}
    '''
    fp = open(filename)
    for i, line in enumerate(fp):
        if i == number:
            fp.close()
            return line.replace('\n', '')
    fp.close()

def saveListOfSentencesToFile(l, filename = 'aux.txt'):
    '''
    Description: Saves a list of sentences to a text file. If the input parameter is not a list of strings, the function does nothing and returns None.\n
    Input: List of sentences to save and name of the file in disk.\n
    Output: None.\n 
    Example:\n
    C{In: l = ['First sentence', 'Second sentence', 'Third sentence']}\n
    C{In: saveListOfSentencesToFile(l)}\n
    C{$cat aux.txt}\n
    C{$First sentence}\n
    C{$Second sentence}\n
    C{$Third sentence}\n    
    Related functions: L{readListOfSentencesFromFile}\n
    @param l: list to save to disk
    @type l: C{list}
    @param filename: file name
    @type filename: C{str}
    @return: C{None}
    @rtype: C{NoneType}
    '''
    if not(all(type(item) is str for item in l)):
        return None
    f = open(filename, 'w')
    for item in l:
        print>>f, item
    f.close()

def readListOfSentencesFromFile(filename):
    '''
    Description: Reads a list of sentences from a text file into a list.\n
    Input: Name of the file from where the list of sentences is read.\n
    Output: List with the read sentences.\n 
    Example:\n
    C{$cat aux.txt}\n
    C{$First sentence}\n
    C{$Second sentence}\n
    C{$Third sentence}\n    
    C{In: readListOfSentencesFromFile('aux.txt')}\n
    C{Out: ['First sentence', 'Second sentence', 'Third sentence']}\n
    Related functions: L{saveListOfSentencesToFile}\n
    @param filename: file name
    @type filename: C{str}
    @return: 
    @rtype: C{list}
    '''
    return [line.rstrip('\n') for line in open(filename)]

def lastWords(sentences):
    '''
    Description: Returns a list with the last words of a list of sentences. If the input parameter is not a list of strings, the function does nothing and returns None.\n
    Input: List of sentences.\n
    Output: List of last words.\n
    Example:\n
    C{In: lastWords(['First sentence', 'Second sentence', 'This is another one'])}\n
    C{Out: ['sentence', 'sentence', 'one']}\n
    Related functions:\n
    @param sentences: list of sentences
    @type sentences: list
    @return:
    @rtype: C{list}
    '''
    if not(all(type(item) is str for item in sentences)):
        return None
    return [sentence.split()[-1].translate(None, '.?=')
    for sentence in sentences]

def sortStringListByReverseString(l):
    '''
    Description: Sorts a list of strings in the order of the reversed strings. Thus, the sorting is made according to the endings.\n
    Input: List of strings.\n
    Output: List of strings ordered by endind.\n
    Example:\n
    C{In: wordList = ['ama', 'donostia', 'soilik', 'aitona', 'bera', 'da', 'figurazioa', 'bakarrik', 'amama']}\n
    C{In: sortStringListByReverseString(wordList)}\n
    C{Out: ['da', 'donostia', 'ama', 'amama', 'aitona', 'figurazioa', 'bera', 'soilik', 'bakarrik']}\n
    Related functions: L{lastWords}\n
    @param l: list of strings
    @type l: C{list}
    @return:
    @rtype: C{list}
    '''
    y = [z[::-1] for z in l]
    return [l for (y, l) in sorted(zip(y, l), key=lambda pair: pair[0])]

def cleanDirtyStrings(s):
    '''
    '''
    aux = s
    aux = aux.replace("u'","")
    aux = aux.replace("',","")
    aux = aux.replace("'","")
    aux = aux.replace(",","")
    aux = aux.replace("[","")
    aux = aux.replace("]","")
    return aux

def cleanDocument(D, M):
    '''
    Description: Returns a document without punctuation marks, without leading or trailing spaces and with all sequences of
    whitespaces reduced to a single whitespace.\n
    Input: The document to clean and the set of punctuation marks.\n
    Output: Cleaned document.\n
    Example:\n
    C{In: document = 'Lehiakortasunerako proiektuetan, gutxienez 100.000 euroko inbertsioa exijituko da, eta izaera estrategikoetan, gutxienez 4 milioi eurokoa.'}\n
    C{In: marks = (',','.','?','!','"',"'",'\','//')}\n
    C{Out: 'Lehiakortasunerako proiektuetan gutxienez 100000 euroko inbertsioa exijituko da eta izaera estrategikoetan gutxienez 4 milioi eurokoa'}\n
    Related functions:\n
    @param D: document to clean
    @type D: C{str}
    @param M: marks to remove
    @type M: C{tuple}
    @return: C{str}
    '''
    without_marks = D.translate(None, ''.join(M))
    return ' '.join(without_marks.split())

def intersection(l1, l2):
    '''
    Description: Returns a set containing the elements common to the two lists. If the intersection is empty, returns the empty set.
    If it is not possible to convert the result to a set, the result is None.\n
    Input: The two lists from where we want to find the intersection.\n
    Output: Set containing the common elements.\n
    Example:\n
    C{In: wordList1 = ['ama', 'donostia', 'soilik', 'aitona', 'bera', 'da', 'figurazioa', 'bakarrik', 'amama', 'soilik']}\n
    C{In: wordList2 = ['aitona', 'soilik', 'eguzkia', 'elurra', 'amama', 'bera', 'da', 'izotza']}\n
    C{In: intersection(wordList1, wordList2)}\n
    C{Out: {'aitona', 'amama', 'bera', 'da', 'soilik'}}\n
    Related functions: L{thereIsIntersection}\n
    @param l1: first list to find the intersection
    @type l1: C{list}
    @param l2: second list to find the intersection
    @type l2: C{list}
    @return:
    @rtype: C{set}
    '''
    intersec = [val for val in l1 if val in l2]
    if not(all(isinstance(item, collections.Hashable) for item in intersec)):
        return None
    return set(intersec)

def thereIsIntersection(l1, l2):
    '''
    Description: Returns C{true} if the intersection exists and it is not empty. Returns {false} otherwise.\n
    Input: The two lists from where we want to find the intersection.\n
    Output: Boolean value telling if there is intersection or not.\n
    Example:\n
    C{In: wordList1 = ['ama', 'donostia', 'soilik', 'aitona', 'bera', 'da', 'figurazioa', 'bakarrik', 'amama', 'soilik']}\n
    C{In: wordList2 = ['aitona', 'soilik', 'eguzkia', 'elurra', 'amama', 'bera', 'da', 'izotza']}\n
    C{In: wordList3 = [['ama', 'donostia'], 'soilik', 'aitona', 'bera', 'da', 'figurazioa']}\n
    C{In: wordList4 = ['soilik', 'aitona', 'bera', 'da', 'figurazioa']}\n
    C{In: wordList5 = ['ura', 'beroa', 'euria']}\n
    C{In: wordList6 = ['soilik', 'aitona', 'bera', 'da', 'figurazioa', ['ama', 'donostia']]}\n
    C{In: thereIsIntersection(wordList1, wordList2)}\n
    C{Out: True}\n
    C{In: thereIsIntersection(wordList1, wordList3)}\n
    C{Out: True}\n
    C{In: thereIsIntersection(wordList4, wordList5)}\n
    C{Out: False}\n
    C{In: thereIsIntersection(wordList3, wordList6)}\n
    C{Out: False}\n
    Related functions: L{intersection}\n
    @param l1: first list to find the intersection
    @type l1: C{list}
    @param l2: second list to find the intersection
    @type l2: C{list}
    @return:
    @rtype: C{boolean}
    '''
    result = intersection(l1, l2)
    return (not result is None and len(result) > 0)

def removeExtraPartitions(l):
    '''
    MIRAR MEJOR
    '''
    return [value for index, value in enumerate(l)
      if len([value2 for ind, value2 in enumerate(l)
      if ind > index and thereIsIntersection(value2, value)]) == 0]

def createHistogram(l, outputFilename, titleText, xAxisText, yAxisText):
    '''
    '''
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1,)
    numBins = int(round(np.sqrt(len(l))))
    n, bins, patches = ax.hist(l, bins=numBins, range=(min(l), max(l)), histtype='bar')
    for patch in patches:
        patch.set_facecolor('r')
    pyplot.title(titleText)
    pyplot.xlabel(xAxisText)
    pyplot.ylabel(yAxisText)
    pyplot.savefig(outputFilename)

def substractList(l1, l2):
    '''
    '''
    return list(set(l1) - set(l2))

def IQR(l):
    '''
    '''
    q75, q25 = np.percentile(l, [75 ,25])
    return q75 - q25

def FreedmanDiaconisValue(l):
    '''
    '''
    return 2 * IQR(l) * pow(len(l), -1.0/3.0)

def numberBins(l, functionToComputeWidth = FreedmanDiaconisValue):
    '''
    '''
    return np.ceil(float((max(l) - min(l))) / FreedmanDiaconisValue(l))
