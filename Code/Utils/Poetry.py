# -*- coding: utf-8 -*-
'''
Created on Apr 7, 2015

@author: José María Martínez-Otzeta

Collection of Python functions used in the Bertsobot project

'''
from gensim import corpora, models, similarities, matutils
import numpy as np
import itertools
import re
import subprocess
import collections
import General as General
import NLP as NLP

def rhymingSentences(sentence, filenameOrList):
    '''
    Returns all the rhyming sentences of I{sentence} in a I{filename}.\n
    Example:\n
    C{rhymingSentences("ilargi", "8tx_nahas_mahas.txt")}\n
    C{['baina espainia ez zen urduri jarri',
    'egunero izango da erabilgarri',
    ...
    'ez dela behar bezain ongi iragarri',
    'Sua zergatik piztu zen ez dago argi']}\n   
    @param sentence: sentence that marks the rhyming pattern
    @type sentence: C{str}
    @param filename: filename with the prospective rhymes
    @type filename: C{str}
    @return: list of strings (sentences)
    @rtype: list
    ''' 
    if isinstance(filenameOrList, list):
        filename = 'aux.txt'
        General.saveListOfSentencesToFile(filenameOrList, filename)
        s = subprocess.check_output(['perl', 'generateRhymeSentences.pl', sentence, filename]) 
    if isinstance(filenameOrList, str):
        s = subprocess.check_output(['perl', 'generateRhymeSentences.pl', sentence, filenameOrList]) 
    s = s.split('\n')
    s = filter(None, s)
    return s

def rhymingIndices(sentence, filenameOrList):
    '''
    Returns the indices (number of line, starting by 1) of all the rhyming sentences of I{sentence} in a I{filename}.\n
    Example:\n
    C{rhymingIndices("ilargi", "8tx_nahas_mahas.txt")}\n
    C{[637, 1637, ..., 52135, 52346]}\n   
    @param sentence: sentence that marks the rhyming pattern
    @type sentence: C{str}
    @param filename: filename with the prospective rhymes
    @type filename: C{str}
    @return: list of integers (indices)
    @rtype: list
    ''' 
    if isinstance(filenameOrList, list):
        filename = 'aux.txt'
        General.saveListOfSentencesToFile(filenameOrList, filename)
        s = subprocess.check_output(['perl', 'generateRhymeIndices.pl', sentence, filename]) 
    if isinstance(filenameOrList, str):
        s = subprocess.check_output(['perl', 'generateRhymeIndices.pl', sentence, filenameOrList]) 
    s = s.split('\n')
    s = filter(None, s)
    return map(int, s)

def noRhymingIndices(filenameOrList):
    '''
    Returns the indices (number of line, starting by 1) of all the sentences in a I{filename} with no rhyming pattern. 
    These are the sentences that have weird endings that have not been included in the rhyming regular expressions.\n
    Example:\n
    C{noRhymingIndices("8tx_nahas_mahas.txt")}\n
    C{[270, 389, ..., 52498, 52658]}\n
    @param filename: filename with the prospective rhymes
    @type filename: C{str}
    @return: list of integers (indices)
    @rtype: list
    ''' 
    if isinstance(filenameOrList, list):
        filename = 'aux.txt'
        General.saveListOfSentencesToFile(filenameOrList, filename)
        s = subprocess.check_output(['perl', 'findNoRhymeIndices.pl', filename]) 
    if isinstance(filenameOrList, str):
        s = subprocess.check_output(['perl', 'findNoRhymeIndices.pl', filenameOrList]) 
    s = s.split('\n')
    s = filter(None, s)
    return map(int, s)

def noRhymingSentences(filenameOrList):
    '''
    Returns the sentences in a I{filename} with no rhyming pattern. 
    These are the sentences that have weird endings that have not been included in the rhyming regular expressions.\n
    Example:\n
    C{noRhymingSentences("8tx_nahas_mahas.txt")}\n
    C{['manifestazioa antolatzeko eajk',
    'hala nola jodie foster eta sam neill',
    ...
    'atzo bilkurak egin zituzten erletxes',
    'oso azkar irtetzen zen erasora rosenborg']}\n   
    @param filename: filename with the prospective rhymes
    @type filename: C{str}
    @return: list of strings (sentences)
    @rtype: list
    ''' 
    if isinstance(filenameOrList, list):
        filename = 'aux.txt'
        General.saveListOfSentencesToFile(filenameOrList, filename)
        s = subprocess.check_output(['perl', 'findNoRhymeSentences.pl', filename]) 
    if isinstance(filenameOrList, str):
        s = subprocess.check_output(['perl', 'findNoRhymeSentences.pl', filenameOrList]) 
    s = s.split('\n')
    s = filter(None, s)
    return s

def noRhymingLastWords(filenameOrList):
    '''
    Returns the last words in the sentences in a I{filename} with no rhyming pattern. 
    These are the sentences that have weird endings that have not been included in the rhyming regular expressions.\n
    Example:\n
    C{noRhymingLastWords("8tx_nahas_mahas.txt")}\n
    C{['eajk', 'neill', ..., 'erletxes', 'rosenborg']}\n   
    @param filename: filename with the prospective rhymes
    @type filename: C{str}
    @return: list of strings (words)
    @rtype: list
    ''' 
    return General.lastWords(noRhymingSentences(filenameOrList))

def getEquivalenceClassForVerse(sentence, partition):
    '''
    '''
    found = False
    i = 0
    equivalence_class = []
    while not found and i < len(partition):
        if len(rhymingSentences(sentence, partition[i])) > 0:
            equivalence_class = list(partition[i])
        i = i + 1 
    return equivalence_class

def getBestStanza(sentence, l):
    '''
    '''
    stanza = list()
    restricted_l = list()
    stanza_fitness = -1000
    # remove sentences with the same ending
    last_words = list()
    last_words.append(sentence.split()[-1])
    for i in range(len(l)):
        if l[i][0].split()[-1] not in last_words:
            restricted_l.append(l[i])
        last_words.append(l[i][0].split()[-1])
    if len(restricted_l) >= 3:
        stanza.append(sentence)
        for j in range(3):
            stanza.append(restricted_l[j][0])
        stanza_fitness = sum([elem[1] for elem in restricted_l[0:2]])
    return stanza, stanza_fitness, restricted_l

def experimentBertsobot(modelsName, theme, filenameLemmatizedPuntu, numberSimilarPuntu, numberBestBertso, filenameOriginalPuntu, functionBestCombinations, functionFitness):
    '''
    Returns the result of a Bertsobot experiment.\n
    Example: C{a, b, c, d  = experimentBertsobot('new', 'aita', '8tx_lem_nahas.txt', 10, 3, 
    '8tx_nahas_mahas.txt', bestCombinationsBruteForce, withOriginalSentenceCoherence)}\n
    @param modelsName: Name of the model stored in disk.
    @type modelsName: C{str}
    @param theme: Sentence for which similarities are going to be computed.
    @type theme: C{str}
    @param filenameLemmatizedPuntu: File with the lemmatized sentences.
    @type filenameLemmatizedPuntu: C{str}
    @param numberSimilarPuntu: Number of similar sentences to I{theme}. The similarity is computed among the lemmatized sentences 
    according to the model recovered from disk.
    @type numberSimilarPuntu: C{int}
    @param filenameOriginalPuntu: File with the original sentences, from where rhyming sentences will be found.
    @type filenameOriginalPuntu: C{str}
    @param functionBestCombinations: Function that returns the best combinations from a set of sentences according to functionFitness. Different functions
    will differ in strategy: brute force, genetic algorithms...
    @type functionBestCombinations: C{function}
    @param functionFitness: Function that returns the fitness of a puntu combination.
    @type functionFitness: C{function}
    @return:
    @rtype: (dict, list, list, list)
    ''' 
    # load the previously computed models 
    dictionary, corpus, tfidfModel, lsiModel = NLP.loadPrecomputedData(modelsName)
    # get the similaraties of all the lines in the lemmatized file with respect to the topic models
    similarityMatrixOfFilenameAccordingToComputedModels = NLP.getSimilarityMatrix(filenameLemmatizedPuntu, dictionary, tfidfModel, lsiModel)
    # get the similarities of the theme with respect to the lemmatized sentences
    simsWithTheme = NLP.simsFromSentence(theme, dictionary, lsiModel, similarityMatrixOfFilenameAccordingToComputedModels)
    # get the lemmatized sentences more similar to theme, along with their indices and values
    bestIndices, bestValues, bestSentencesLemmatized, bestSentencesOriginal = NLP.getIndexesAndSentencesFromSimsValues(simsWithTheme, filenameLemmatizedPuntu, filenameOriginalPuntu, numberSimilarPuntu)
    
    #removeSentencesFromListWithSameEndWordThatModel(model, sentences):
    rhymingSent = [rhymingSentences(x, filenameOriginalPuntu) for x in bestSentencesOriginal]
    rhymingInd = [rhymingIndices(x, filenameOriginalPuntu) for x in bestSentencesOriginal]
    rhymingSentClean = [removeSentencesFromListWithSameEndWordThatModel(x[1],x[0]) for x in zip(rhymingSent, bestSentencesOriginal)]
    rhymingIndClean = [[y[0] for y in zip(x[0],x[1]) if not checkSentenceSameEndWordThatModel(x[2], y[1])] 
                       for x in zip(rhymingInd, rhymingSent, bestSentencesOriginal)]
    # the previous instructions have removed the bestSentecesOriginal from rhymingSentClean; now we need to add again those sentences and indices to the list
    [x[0].append(x[1]) for x in zip(rhymingSentClean, bestSentencesOriginal)]
    [x[0].append(x[1]) for x in zip(rhymingIndClean, bestIndices)]
    #print rhymingSentClean
    #print bestSentencesOriginal
    # get a dictionary of best sentences to rhymes indices in the original file
    dictSentencesToIndicesRhymes = {x[0]:(x[1], x[2], x[3], x[4]) for x in zip(bestSentencesOriginal, rhymingIndClean, rhymingSentClean, bestValues, bestIndices)}   
    bestBertsos = [(k, General.readLineNumberFromFile(v[3], filenameOriginalPuntu), 
                     [(elem[0] + [General.readLineNumberFromFile(v[3], filenameOriginalPuntu)], elem[1],elem[2]) 
                     for elem in functionBestCombinations(theme, k, v[0], filenameLemmatizedPuntu, filenameOriginalPuntu, 
                                            similarityMatrixOfFilenameAccordingToComputedModels, numberBestBertso, 
                                            dictionary, tfidfModel, lsiModel, functionFitness)]) for k, v in dictSentencesToIndicesRhymes.items()]
    return dictSentencesToIndicesRhymes, bestSentencesOriginal, bestIndices, bestBertsos

def checkSentenceSameEndWordThatModel(model, sentence):
    '''
    TODO\n
    @param model: model
    @type model: C{str}
    @param sentences: sentences
    @type sentences: C{list}
    @return:
    @rtype: C{list}
    '''
    return sentence.split()[-1].translate(None, '.?=')==model.split()[-1].translate(None, '.?=')

def removeSentencesFromListWithSameEndWordThatModel(model, sentences):
    '''
    TODO\n
    @param model: model
    @type model: C{str}
    @param sentences: sentences
    @type sentences: C{list}
    @return:
    @rtype: C{list}
    '''
    return [sentence for sentence in sentences if sentence.split()[-1].translate(None, '.?=')!=model.split()[-1].translate(None, '.?=')]

def validBertso(bertso):
    '''
    Returns if a bertso is valid, this is, no repeated words in the end of the lines.\n
    Example:\n
    C{validBertso(['a b','a c','c b','f g'])}
    C{False}
    C{validBertso(['a b','a c','c n','f g'])}
    C{True}
    @param bertso: bertso
    @type bertso: list
    @return:
    @rtype: C{boolean}
    '''
    # list of lists. No repetitions among last words
    lastW = General.lastWords(bertso)
    return len(lastW)==len(set(lastW))

def internalCoherence(setIndexes, theme, originalSentence, filenameLemmatizedPuntu, simsMatrix, dictionary, tfidfModel, lsiModel):
    '''
    Returns if a bertso is valid, this is, no repeated words in the end of the lines.\n
    Example:\n
    C{internalCoherence}
    @param simsMatrix: simsMatrix
    @type simsMatrix: C{numpy.ndarray}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param tfidfModel: tfidfModel
    @type tfidfModel: C{models.tfidfmodel.TfidfModel}
    @param lsiModel: lsiModel
    @type lsiModel: C{models.lsimodel.LsiModel}
    @return:
    @rtype: C{float}
    '''
    setInd = [x-1 for x in setIndexes]
    setSentences = [General.readLineNumberFromFile(x, filenameLemmatizedPuntu) for x in setInd]
    sims = NLP.similarityAmongSetSentences(setSentences, NLP.cosineSimilarityBetweenTwoSentences, dictionary, tfidfModel, lsiModel)
    arr, arrMin, arrMax, arrMean, arrStd, arrMedian = NLP.getStatisticsFromSimilarityMatrix(sims)
    return arrMedian

def withOriginalSentenceCoherence(setIndexes, theme, originalSentence, filenameLemmatizedPuntu, simsMatrix, dictionary, tfidfModel, lsiModel):
    '''
    Returns if a bertso is valid, this is, no repeated words in the end of the lines.\n
    Example:\n
    C{internalCoherence}
    @param simsMatrix: simsMatrix
    @type simsMatrix: C{numpy.ndarray}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param tfidfModel: tfidfModel
    @type tfidfModel: C{models.tfidfmodel.TfidfModel}
    @param lsiModel: lsiModel
    @type lsiModel: C{models.lsimodel.LsiModel}
    @return:
    @rtype: C{float}
    '''
    setInd = [x-1 for x in setIndexes]
    setSentences = [General.readLineNumberFromFile(x, filenameLemmatizedPuntu) for x in setInd]
    sims = [NLP.cosineSimilarityBetweenTwoSentences(originalSentence, x, dictionary, tfidfModel, lsiModel) for x in setSentences]
    return np.mean(sims)

def withThemeCoherence(setIndexes, theme, originalSentence, filenameLemmatizedPuntu, simsMatrix, dictionary, tfidfModel, lsiModel):
    '''
    Returns if a bertso is valid, this is, no repeated words in the end of the lines.\n
    Example:\n
    C{internalCoherence}
    @param simsMatrix: simsMatrix
    @type simsMatrix: C{numpy.ndarray}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param tfidfModel: tfidfModel
    @type tfidfModel: C{models.tfidfmodel.TfidfModel}
    @param lsiModel: lsiModel
    @type lsiModel: C{models.lsimodel.LsiModel}
    @return:
    @rtype: C{float}
    '''
    setInd = [x-1 for x in setIndexes]
    setSentences = [General.readLineNumberFromFile(x, filenameLemmatizedPuntu) for x in setInd]
    sims = [NLP.cosineSimilarityBetweenTwoSentences(theme, x, dictionary, tfidfModel, lsiModel) for x in setSentences]
    return np.mean(sims)

def bestCombinationsLookingRhymes(theme, originalSentence, bestIndices, filenameLemmatizedPuntu, filenameOriginalPuntu, simsMatrix, numberResults, dictionary, tfidfModel, lsiModel, functionFitness):
    '''
    Returns the best combination of puntus from a list of I{bestIndices}, according to a I{simsMatrix} matrix.\n
    Example:\n
    @param theme: theme
    @type theme: C{str}
    @param originalSentence: originalSentence
    @type originalSentence: C{str}
    @param bestIndices: bestIndices
    @type bestIndices: list
    @param filenameLemmatizedPuntu: filenameLemmatizedPuntu
    @type filenameLemmatizedPuntu: C{str}    
    @param filenameOriginalPuntu: filenameOriginalPuntu
    @type filenameOriginalPuntu: C{str}
    @param simsMatrix: simsMatrix
    @type simsMatrix: C{numpy.ndarray}
    @param numberResults: numberResults
    @type numberResults: C{int}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param tfidfModel: tfidfModel
    @type tfidfModel: C{models.tfidfmodel.TfidfModel}
    @param lsiModel: lsiModel
    @type lsiModel: C{models.lsimodel.LsiModel}
    @param functionFitness: functionFitness
    @type functionFitness: C{function}
    @return:
    @rtype: list
    '''
    # TODO
    numberNeededPuntu = 3
    # bestResults has the bertso, the indices and the value
    bestInd = [x-1 for x in bestIndices]
    allRhymes = [General.readLineNumberFromFile(x, filenameOriginalPuntu) for x in bestInd]
    allRhymesLemmatized = NLP.lemmatizeListOfSentences(allRhymes, 'aux.txt')
    sims = NLP.similarityFromSentenceToSetSentences(theme, allRhymesLemmatized, NLP.cosineSimilarityBetweenTwoSentences, dictionary, tfidfModel, lsiModel)
    results = sorted(zip(allRhymes, bestInd, sims), key=lambda pair: pair[2], reverse = True)
    bestResults = [([General.readLineNumberFromFile(x, filenameOriginalPuntu) for x in combination], combination, 
                    functionFitness(combination, theme, originalSentence, filenameLemmatizedPuntu, simsMatrix, dictionary, tfidfModel, lsiModel)) 
                   for combination in itertools.combinations(bestInd, numberNeededPuntu)]
    if bestResults:
        bestResults = [x for x in bestResults if validBertso(x[0])]
        bestResults.sort(key=lambda x: x[2], reverse = True)
    return bestResults[0:numberResults]
    
def bestCombinationsBruteForce(theme, originalSentence, bestIndices, filenameLemmatizedPuntu, filenameOriginalPuntu, simsMatrix, numberResults, dictionary, tfidfModel, lsiModel, functionFitness):
    '''
    Returns the best combination of puntus from a list of I{bestIndices}, according to a I{simsMatrix} matrix.\n
    Example:\n
    @param theme: theme
    @type theme: C{str}
    @param originalSentence: originalSentence
    @type originalSentence: C{str}
    @param bestIndices: bestIndices
    @type bestIndices: list
    @param filenameLemmatizedPuntu: filenameLemmatizedPuntu
    @type filenameLemmatizedPuntu: C{str}    
    @param filenameOriginalPuntu: filenameOriginalPuntu
    @type filenameOriginalPuntu: C{str}
    @param simsMatrix: simsMatrix
    @type simsMatrix: C{numpy.ndarray}
    @param numberResults: numberResults
    @type numberResults: C{int}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param tfidfModel: tfidfModel
    @type tfidfModel: C{models.tfidfmodel.TfidfModel}
    @param lsiModel: lsiModel
    @type lsiModel: C{models.lsimodel.LsiModel}
    @param functionFitness: functionFitness
    @type functionFitness: C{function}
    @return:
    @rtype: list
    '''
    numberNeededPuntu = 3
    # bestResults has the bertso, the indices and the value
    bestInd = [x-1 for x in bestIndices]
    bestResults = [([General.readLineNumberFromFile(x, filenameOriginalPuntu) for x in combination], combination, 
                    functionFitness(combination, theme, originalSentence, filenameLemmatizedPuntu, simsMatrix, dictionary, tfidfModel, lsiModel)) 
                   for combination in itertools.combinations(bestInd, numberNeededPuntu)]
    if bestResults:
        bestResults = [x for x in bestResults if validBertso(x[0])]
        bestResults.sort(key=lambda x: x[2], reverse = True)
    print bestResults
    return bestResults[0:numberResults]

# def bestCombinationsGreedyTheme(theme, filenameLemmatizedPuntu, filenameOriginalPuntu, simsMatrix, numberResults, dictionary, tfidfModel, lsiModel):
#     '''
#     Returns the best combination of puntus from a list of I{bestIndices}, according to a I{simsMatrix} matrix.\n
#     Example:\n
#     @param theme: theme
#     @type theme: C{str}
#     @param originalSentence: originalSentence
#     @type originalSentence: C{str}
#     @param bestIndices: bestIndices
#     @type bestIndices: list
#     @param filenameLemmatizedPuntu: filenameLemmatizedPuntu
#     @type filenameLemmatizedPuntu: C{str}    
#     @param filenameOriginalPuntu: filenameOriginalPuntu
#     @type filenameOriginalPuntu: C{str}
#     @param simsMatrix: simsMatrix
#     @type simsMatrix: C{numpy.ndarray}
#     @param numberResults: numberResults
#     @type numberResults: C{int}
#     @param dictionary: dictionary
#     @type dictionary: C{corpora.dictionary.Dictionary}
#     @param tfidfModel: tfidfModel
#     @type tfidfModel: C{models.tfidfmodel.TfidfModel}
#     @param lsiModel: lsiModel
#     @type lsiModel: C{models.lsimodel.LsiModel}
#     @param functionFitness: functionFitness
#     @type functionFitness: C{function}
#     @return:
#     @rtype: list
#     '''
#     tentativeList = list()
#     withThemeList = 
#     numberNeededPuntu = 3
#     # bestResults has the bertso, the indices and the value
#     bestInd = [x-1 for x in bestIndices]
#     bestResults = [([General.readLineNumberFromFile(x, filenameOriginalPuntu) for x in combination], combination, 
#                     functionFitness(combination, theme, originalSentence, filenameLemmatizedPuntu, simsMatrix, dictionary, tfidfModel, lsiModel)) 
#                    for combination in itertools.combinations(bestInd, numberNeededPuntu)]
#     if bestResults:
#         bestResults = [x for x in bestResults if validBertso(x[0])]
#         bestResults.sort(key=lambda x: x[2], reverse = True)
#     print bestResults
#     return bestResults[0:numberResults]

# def rhymingPartition(filename):
#     '''
#     Returns the best combination of puntus from a list of I{bestIndices}, according to a I{simsMatrix} matrix.\n
#     Example:\n
#     @param functionFitness: functionFitness
#     @type functionFitness: C{function}
#     @return:
#     @rtype: list
#     '''
#     partitionIndices = []
#     partitionSentences = []
#     num_lines = sum(1 for line in open(filename))
#     fp = open(filename)
#     for i, line in enumerate(fp):
#         sentence = line.replace('\n','')
#         found = any([sentence in subset for subset in partitionSentences])
#         if (not found):
#             rhyms = rhymingSentences(sentence, filename)
#             inds = rhymingIndices(sentence, filename)
#             if (not rhyms or len(rhyms) > num_lines * 0.8):
#                 partitionSentences.append([sentence])
#                 partitionIndices.append([i])
#             else:
#                 partitionSentences.append(rhyms)
#                 partitionIndices.append(inds)
#     fp.close()
#     return partitionIndices, partitionSentences

def possiblePartitions(partitions, stanzaPattern):
    '''
    Remove the partitions from where it is impossible to build rhymes.
    stanzaPattern is in the form (0,0,0,0), (0,1,0,1,0,1)...
    a partition's length has to be greater or equal than the minimum of verses with the same rhyme. 
    For example, with (0,0,0,0), the minimum is 4. With (0,1,0,1,0,1) the minimum is 3. With (0,1,0,1,0,1,2,2) the minimum is 2. 
    '''
    counter = collections.Counter(stanzaPattern)
    minimumPartitionSize = min(counter.values())
    return [partition for partition in partitions if len(partition) >= minimumPartitionSize]


def rhymingPartition(filenameOrList):
    '''
    Returns a partition of the sentences in filename according to their rhyme.\n
    Example:\n
    @param filename: file with the sentences
    @type filename: C{str}
    @return: list of subsets with common rhyme
    @rtype: list
    '''
    filename = ''
    if isinstance(filenameOrList, list):
        filename = 'aux.txt'
        General.saveListOfSentencesToFile(filenameOrList, filename)
    if isinstance(filenameOrList, str):
        filename = filenameOrList
    
    partitionIndices = []
    partitionSentences = []
    alreadyFound = []
    num_lines = sum(1 for line in open(filename))
    noRhyme = sorted(noRhymingIndices(filename))
    fp = open(filename)
    # i starts in zero
    for i, line in enumerate(fp):
        # print i
        if (i+1 not in noRhyme and i+1 not in alreadyFound):
            sentence = line.replace('\n','')
            rhyms = rhymingSentences(sentence, filename)
            inds = rhymingIndices(sentence, filename)
            if (not rhyms or len(rhyms) > num_lines * 0.8):
                partitionSentences.append([sentence])
                partitionIndices.append([i])
                alreadyFound = sorted(alreadyFound + [i])
            else:
                partitionSentences.append(rhyms)
                partitionIndices.append(inds)
                alreadyFound = sorted(alreadyFound + inds)
    fp.close()
    
    return partitionIndices, partitionSentences

#def analyzeProspectiveRhymes(filename, histogramFilename):
def analyzeProspectiveRhymes(filenameOrList):
    '''
    Analyzes a set of prospective rhymes, returning the sentences without rhyming pattern, the rhyming partitions and the histograms.\n
    Example:\n
    @param filenameOrList: filename or list
    @type filenameOrList: C{str} or C{list}
    @return:
    @rtype: C{list, list}
    '''   
    noRhymeSentences = sorted(noRhymingSentences(filenameOrList))
    noRhymeLastWords = General.sortStringListByReverseString(noRhymingLastWords(filenameOrList))
    partitionIndices, partitionSentences = rhymingPartition(filenameOrList)
    cleanPartitionIndices = General.removeExtraPartitions(partitionIndices)
    cleanPartitionSentences = General.removeExtraPartitions(partitionSentences)
    #lengthsPartitionIndices = [len(elem) for elem in cleanPartitionIndices]
    #General.createHistogram(lengthsPartitionIndices, histogramFilename + '.png', 'Distribution of number of sentences', 'Number of sentences in subset', 'Number of subsets') 
    #General.createHistogram(np.log(lengthsPartitionIndices), histogramFilename + 'Log.png', 'Distribution of number of sentences', 'Number of sentences in subset (log)', 'Number of subsets') 
    #lengthsPartitionIndices1To10 = [elem for elem in lengthsPartitionIndices if elem >= 1 and elem <= 25]
    #General.createHistogram(lengthsPartitionIndices1To10, histogramFilename + '1To10.png', 'Distribution of number of sentences', 'Number of sentences in subset', 'Number of subsets')                                           
    #General.createHistogram(np.log(lengthsPartitionIndices1To10), histogramFilename + 'Log1To10.png', 'Distribution of number of sentences', 'Number of sentences in subset (log)', 'Number of subsets')
    #lengthsPartitionIndicesMoreThan10 = [elem for elem in lengthsPartitionIndices if elem > 25]
    #General.createHistogram(lengthsPartitionIndicesMoreThan10, histogramFilename + 'MoreThan10.png', 'Distribution of number of sentences', 'Number of sentences in subset', 'Number of subsets')                                           
    #General.createHistogram(np.log(lengthsPartitionIndicesMoreThan10), histogramFilename + 'LogMoreThan10.png', 'Distribution of number of sentences', 'Number of sentences in subset (log)', 'Number of subsets')                                                                                      
    return noRhymeSentences, noRhymeLastWords, cleanPartitionIndices, cleanPartitionSentences

def removeBadPatterns(listPatterns, listSentences):
    '''
    Analyzes a set of prospective rhymes, returning the sentences without rhyming pattern, the rhyming partitions and the histograms.\n
    Example:\n
    l, m = removeBadPatterns(['.*[cqvwy]|ng|ss.*'], w)
    @param listPatterns
    @type filename: C{str}
    @return:
    @rtype: C{list, list}
    '''   
    listRe = [re.compile(pattern) for pattern in listPatterns]
    badSentences = [sentence for sentence in listSentences if any([item.search(sentence) for item in listRe]) is not None]
    goodSentences = [sentence for sentence in listSentences if any([item.search(sentence) for item in listRe]) is None]
    return badSentences, goodSentences
