# -*- coding: utf-8 -*-
'''
Created on Apr 7, 2015

@author: José María Martínez-Otzeta

Collection of Python functions used in the Bertsobot project

'''
from gensim import corpora, models, similarities, matutils
import numpy as np
import subprocess
import General as General
import Customize

def loadPrecomputedData(name):
    '''
    The output variables are assigned the values of previously saved files denoted by I{name}.\n
    Example:\n
    C{myDict, myMmCorpus, myTfidfModel, myLsiModel = loadPrecomputedData("example")}\n
    The variables I{myDict}, I{myMmCorpus}, I{myTfidfModel}, I{myLsiModel} are assigned the contents of the previously saved files
    I{example.dict}, I{example.mm}, I{example.tfidf} and I{example.lsi}.\n
    Related functions: L{savePrecomputedData}.
    @param name: filename
    @type name: C{str}
    @return:
    @rtype: C{(corpora.dictionary.Dictionary, corpora.mmcorpus.MmCorpus, models.tfidfmodel.TfidfModel, models.lsimodel.LsiModel)}
    '''
    return corpora.Dictionary.load(name + '.dict'), corpora.MmCorpus(name + '.mm'), models.TfidfModel.load(name + '.tfidf'), models.LsiModel.load(name + '.lsi')

def savePrecomputedData(dictionary, corp_bow, tfidf, lsi, name):
    '''
    Stores the input variables in files denoted by I{name}.\n
    Example:\n
    C{savePrecomputedData(myDict, myMmCorpus, myTfidfModel, myLsiModel, "example")}\n
    The variables I{myDict}, I{myMmCorpus}, I{myTfidfModel}, I{myLsiModel} are stored in the files
    I{example.dict}, I{example.mm}, I{example.tfidf} and I{example.lsi}.\n
    Related functions: L{loadPrecomputedData}.
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param corp_bow: corp_bow
    @type corp_bow: C{corpora.mmcorpus.MmCorpus} or C{list}
    @param tfidf: tfidf
    @type tfidf: C{models.tfidfmodel.TfidfModel}
    @param lsi: lsi
    @type lsi: C{models.lsimodel.LsiModel}
    @param name: name
    @type name: C{str}
    '''
    dictionary.save(name + '.dict')
    corpora.MmCorpus.serialize(name + '.mm', corp_bow)
    tfidf.save(name + '.tfidf')
    lsi.save(name + '.lsi')

def computeData(texts, num_topics):
    '''
    Accepts a list of lists of strings and a integer as parameters, and returns four variables of type I{corpora.dictionary.Dictionary}, I{list},
    I{models.tfidfmodel.TfidfModel} and I{models.lsimodel.LsiModel}. The parameter I{texts} contains a corpus and I{num_topics} is the number of topics
    created in the lsi model.\n
    Example:\n
    C{docs = [['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']]}\n
    C{dictionaryDocs, corpusDocs, tfidfModelDocs, lsiModelDocs = computeData(docs, 2)}\n
    The corpus returned in I{corpusDocs} would be the following:\n
    C{[[(0, 1), (1, 1), (2, 1)],
    [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
    [(2, 1), (5, 1), (7, 1), (8, 1)],
    [(1, 1), (5, 2), (8, 1)],
    [(3, 1), (6, 1), (7, 1)],
    [(9, 1)],
    [(9, 1), (10, 1)],
    [(9, 1), (10, 1), (11, 1)],
    [(4, 1), (10, 1), (11, 1)]]}\n
    Related functions: L{computePrunedData}.
    @param texts: texts
    @type texts: C{list}
    @param num_topics: num_topics
    @type num_topics: C{int}
    @return:
    @rtype: C{(corpora.dictionary.Dictionary, corpora.mmcorpus.MmCorpus, models.tfidfmodel.TfidfModel, models.lsimodel.LsiModel)}
    '''
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    return dictionary, corpus, tfidf, lsi

def computePrunedData(texts, no_below, no_above, badWords, num_topics):
    '''
    Similar to the function L{computeData}, but with the restrictions that words in the I{badWords} list, or in less documents than
    I{no_below}, or in more documents than I{no_above} percentage, are deleted.\n
    Example:\n
    C{docs = [['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']]}\n
    C{badWords = ['system', 'minors']}\n
    C{no_below = 2}\n
    C{no_above = 0.5}\n
    C{dictionaryDocs, corpusDocs, tfidfModelDocs, lsiModelDocs = computePrunedData(docs, no_below, no_above, badWords, 2)}\n
    The corpus returned in *corpusDocs* would have this format:\n
    C{[[(5, 1), (8, 1), (10, 1)],
    [(5, 1), (6, 1), (7, 1), (9, 1), (11, 1)],
    [(4, 1), (7, 1), (10, 1)],
    [(4, 1), (8, 1)],
    [(7, 1), (9, 1), (11, 1)],
    [(3, 1)],
    [(1, 1), (3, 1)],
    [(1, 1), (3, 1)],
    [(1, 1), (6, 1)]]}\n
    Related functions: L{computeData}.
    @param texts: texts
    @type texts: C{list}
    @param no_below: no_below
    @type no_below: C{int}
    @param no_above: no_above
    @type no_above: C{float}
    @param badWords: badWords
    @type badWords: C{list}
    @param num_topics: num_topics
    @type num_topics: C{int}
    @return:
    @rtype: C{(corpora.dictionary.Dictionary, corpora.mmcorpus.MmCorpus, models.tfidfmodel.TfidfModel, models.lsimodel.LsiModel)}
    '''
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    badIds = [dictionary.token2id[x] for x in badWords if x in dictionary.values()]
    dictionary.filter_tokens(bad_ids=badIds)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    return dictionary, corpus, tfidf, lsi

def semanticsExtractor(documents, numTopics, filteredWords, noBelow, noAbove):
    myListOfWords = [elem.split() for elem in documents]
    return computePrunedData(myListOfWords, noBelow, noAbove, filteredWords, numTopics)

def tokenizeDocumentList(documents, stopWordsList, countWords):
    '''
    Tokenizes the I{documents} parameter, removing the words in I{stopWordsList}
    and those words that appear less or equal than I{countWords} times.\n
    Example:\n
    C{documents = ["Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"]}\n
    C{stopString = 'for a of and in to the'}\n
    C{stopWordsList = stopString.lower().split()}\n
    C{tokenizeDocumentList(documents, stopWordsList, 1)}\n
    C{[['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']]}\n
    Related functions: L{splitSentences}
    @param documents: documents
    @type documents: C{list}
    @param stopWordsList: stopWordsList
    @type stopWordsList: C{list}
    @param countWords: countWords
    @type countWords: C{int}
    @returns:
    @rtype: C{list}
    '''
    texts = [[word for word in document.lower().split() if word not in stopWordsList]
             for document in documents]
    all_tokens = [item for sublist in texts for item in sublist]
    tokens_few_times = set(word for word in set(all_tokens) if all_tokens.count(word) <= countWords)
    texts = [[word for word in text if word not in tokens_few_times]
         for text in texts]
    return texts

def splitSentences(documents):
    '''
    Splits the list elements into their constituent words.\n
    Example:\n
    C{documents = ["Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"]}\n
    C{splitSentences(documents)}\n
    C{[['human', 'machine', 'interface', 'for', 'lab', 'abc', 'computer', 'applications'],*
    ['a', 'survey', 'of', 'user', 'opinion', 'of', 'computer', 'system', 'response', 'time'],
    ['the', 'eps', 'user', 'interface', 'management', 'system'],
    ['system', 'and', 'human', 'system', 'engineering', 'testing', 'of', 'eps'],
    ['relation', 'of', 'user', 'perceived', 'response', 'time', 'to', 'error', 'measurement'],
    ['the', 'generation', 'of', 'random', 'binary', 'unordered', 'trees'],
    ['the', 'intersection', 'graph', 'of', 'paths', 'in', 'trees'],
    ['graph', 'minors', 'iv', 'widths', 'of', 'trees', 'and', 'well', 'quasi', 'ordering'],
    ['graph', 'minors', 'a', 'survey']]}\n
    Related functions:\n
    @param documents: documents
    @type documents: C{list}
    @return:
    @rtype: C{list}
    '''
    return [[word for word in document.lower().split()] for document in documents]

def buildCorpusFromFileOrListAndDictionary(filenameOrList, dictionary):
    '''
    Builds a corpus from a file (denoted by its name) or from a list of documents.\n
    Example:\n
    C{documents = ["Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"]}\n
    C{docs = tokenizeDocumentList(documents, [], 0)}\n
    C{dictionary = corpora.Dictionary(docs)}\n
    C{corpus = buildCorpusFromFileOrListAndDictionary(documents, dictionary)}\n
    C{[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
    [(2, 1), (8, 1), (9, 2), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1)],
    [(5, 1), (13, 1), (15, 1), (16, 1), (17, 1), (18, 1)],
    [(4, 1), (9, 1), (13, 2), (16, 1), (19, 1), (20, 1), (21, 1)],
    [(9, 1), (11, 1), (14, 1), (15, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1)],
    [(9, 1), (18, 1), (27, 1), (28, 1), (29, 1), (30, 1), (31, 1)],
    [(9, 1), (18, 1), (30, 1), (32, 1), (33, 1), (34, 1), (35, 1)],
    [(9, 1), (19, 1), (30, 1), (32, 1), (36, 1), (37, 1), (38, 1), (39, 1), (40, 1), (41, 1)],
    [(8, 1), (12, 1), (32, 1), (37, 1)]]}\n
    Related functions:\n
    @param filenameOrList:filenameOrList
    @type filenameOrList: C{str} or C{list}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @return:
    @rtype: C{list}
    '''
    if isinstance(filenameOrList, str):
        f = open(filenameOrList)
        corpus = [dictionary.doc2bow(line.lower().split()) for line in f]
        f.close()
    if isinstance(filenameOrList, list):
        corpus = [dictionary.doc2bow(line.lower().split()) for line in filenameOrList]
    return corpus

def getSimilarityMatrix(filenameOrList, dictionary, tfidfModel, lsiModel):
    '''
    Returns the similarity matrix of the documents in the input file or list, according to the dictionary and tfidf and lsi models also
    passed as input. The similarity matrix is an m x n matrix where m is the number of documents in filenameOrList and n is the number of topics
    in lsiModel.\n
    Example:\n
    C{documents = ["Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"]}\n
    C{docs = tokenizeDocumentList(documents, [], 0)}\n
    C{dictionary, corpus, tfidfModel, lsiModel = computeData(docs, 2)}\n
    C{sims = getSimilarityMatrix(documents, dictionary, tfidfModel, lsiModel)}\n
    C{sims.index}\n
    C{array([[ 0.56419468, -0.82564181],
    [ 0.98815131,  0.15348279],
    [ 0.66273677, -0.74885243],
    [ 0.66963506, -0.74269032],
    [ 0.9982208 , -0.05962535],
    [ 0.97897345,  0.20398766],
    [ 0.75438434,  0.65643299],
    [ 0.62894249,  0.77745187],
    [ 0.69296259,  0.72097349]], dtype=float32)}\n
    Related functions:\n
    @param filenameOrList: filenameOrList
    @type filenameOrList: C{str} or C{list}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param tfidfModel: tfidfModel
    @type tfidfModel: C{models.tfidfmodel.TfidfModel}
    @param lsiModel: lsiModel
    @type lsiModel: C{models.lsimodel.LsiModel}
    @return:
    @rtype: C{similarities.docsim.MatrixSimilarity}
    '''
    corpusCurrent = buildCorpusFromFileOrListAndDictionary(filenameOrList, dictionary)
    tfidfCurrent = tfidfModel[corpusCurrent]
    similarityCurrent = similarities.MatrixSimilarity(lsiModel[tfidfCurrent])
    return similarityCurrent

def simsFromSentence(sentence, dictionary, lsiModel, simMatrix):
    '''
    Returns the similarity of a sentence to other sentences, according to a dictionary,
    lsiModel and similarity matrix obtained from those sentences.\n
    Example:\n
    C{documents = ["Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"]}\n
    C{docs = tokenizeDocumentList(documents, [], 0)}\n
    C{dictionary, corpus, tfidfModel, lsiModel = computeData(docs, 2)}\n
    C{sims = getSimilarityMatrix(documents, dictionary, tfidfModel, lsiModel)}\n
    C{s = simsFromSentence("the graph",dictionary, lsiModel, sims)}\n
    C{array([ 0.08719606,  0.93657887,  0.21072534,  0.21975829,  0.84092212,
    0.95332867,  0.97927821,  0.92923713,  0.95736557], dtype=float32)}\n
    Related functions:\n
    @param sentence: sentence
    @type sentence: C{str}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param lsiModel: lsiModel
    @type lsiModel: C{models.lsimodel.LsiModel}
    @param simMatrix: simMatrix
    @type simMatrix: C{similarities.docsim.MatrixSimilarity}
    @return:
    @rtype: C{numpy.ndarray}
    '''
    vec_bow = dictionary.doc2bow(sentence.lower().split())
    vec_lsi = lsiModel[vec_bow] # convert the query to LSI space
    sims = simMatrix[vec_lsi]
    return sims

def similarityAmongSetSentences(sentencesSet, similarityFunction, dictionary, tfidfModel, lsiModel):
    '''
    @param sentencesSet: sentencesSet
    @type sentencesSet: C{list}
    @param similarityFunction: similarityFunction
    @type similarityFunction: C{function}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param tfidfModel: tfidfModel
    @type tfidfModel: C{models.tfidfmodel.TfidfModel}
    @param lsiModel: lsiModel
    @type lsiModel: C{models.lsimodel.LsiModel}
    @return:
    @rtype: C{float}
    '''
    return [[similarityFunction(x, y, dictionary, tfidfModel, lsiModel) for y in sentencesSet] for x in sentencesSet]

def similarityFromSentenceToSetSentences(sentence, sentencesSet, similarityFunction, dictionary, tfidfModel, lsiModel):
    '''
    @param sentencesSet: sentencesSet
    @type sentencesSet: C{list}
    @param similarityFunction: similarityFunction
    @type similarityFunction: C{function}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param tfidfModel: tfidfModel
    @type tfidfModel: C{models.tfidfmodel.TfidfModel}
    @param lsiModel: lsiModel
    @type lsiModel: C{models.lsimodel.LsiModel}
    @return:
    @rtype: C{float}
    '''
    return [similarityFunction(x, sentence, dictionary, tfidfModel, lsiModel) for x in sentencesSet]

def cosineSimilarityBetweenTwoSentences(sentence1, sentence2, dictionary, tfidfModel, lsiModel):
    '''
    @param sentence1: sentence1
    @type sentence1: C{str}
    @param sentence2: sentence2
    @type sentence2: C{str}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param tfidfModel: tfidfModel
    @type tfidfModel: C{models.tfidfmodel.TfidfModel}
    @param lsiModel: lsiModel
    @type lsiModel: C{models.lsimodel.LsiModel}
    @return:
    @rtype: C{float}
    '''
    vec_bow1 = dictionary.doc2bow(sentence1.lower().split())
    vec_lda1 = lsiModel[vec_bow1]
    vec_bow2 = dictionary.doc2bow(sentence2.lower().split())
    vec_lda2 = lsiModel[vec_bow2]
    return matutils.cossim(vec_lda1, vec_lda2)

def hellingerDistanceBetweenTwoSentences(sentence1, sentence2, dictionary, tfidfModel, ldaModel):
    '''
    @param sentence1: sentence1
    @type sentence1: C{str}
    @param sentence2: sentence2
    @type sentence2: C{str}
    @param dictionary: dictionary
    @type dictionary: C{corpora.dictionary.Dictionary}
    @param tfidfModel: tfidfModel
    @type tfidfModel: C{models.tfidfmodel.TfidfModel}
    @param ldaModel: ldaModel
    @type ldaModel: C{models.ldamodel.LdaModel}
    @return:
    @rtype: C{float}
    '''
    vec_bow1 = dictionary.doc2bow(sentence1.lower().split())
    vec_lda1 = ldaModel[vec_bow1]
    vec_bow2 = dictionary.doc2bow(sentence2.lower().split())
    vec_lda2 = ldaModel[vec_bow2]
    dense1 = matutils.sparse2full(vec_lda1, ldaModel.num_topics)
    dense2 = matutils.sparse2full(vec_lda2, ldaModel.num_topics)
    return np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2))**2).sum())

def saveSimilarityMatrixAsCSV(sim, filename):
    '''
    Exports a similarity matrix to CSV format. Admits a I{numpy.ndarray} as parameter.
    If I{similarities.docsim.MatrixSimilarity} are used, remember that the needed I{numpy.ndarray}
    is the field I{similarities.docsim.MatrixSimilarity.index}.\n
    Example:\n
    C{saveSimmilarityMatrixAsCSV(s,"test.csv")}\n
    C{saveSimmilarityMatrixAsCSV(sims.index,"test.csv")}\n
    Related functions:\n
    @param sim: sim
    @type sim: C{numpy.ndarray}
    @param filename: filename
    @type filename: C{str}
    @return: None
    '''
    np.savetxt(filename, np.array([r for r in sim]), delimiter=",")

def getIndexesAndSentencesFromSimsValues(sims, filenameLemmatized, filenameOriginal, numberChosen):
    '''
    Returns the list of I{numberChosen} sentences more similar according to a I{sims} similarity matrix previously computed. Also returns the similarity
    values and their indices in I{filenameLemmatized}
    Example:\n
    Related functions:\n
    @param sims: sims
    @type sims: C{numpy.ndarray}
    @param filenameLemmatized: filenameLemmatized
    @type filenameLemmatized: C{str}
    @param numberChosen: numberChosen
    @type numberChosen: C{int}
    @return: Three lists: list of indices in the file I{filenameLemmatized}, list of similarity values and list of sentences.
    @rtype: C{(list, list, list)}
    '''
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    bestIndexes = []
    bestValues = []
    bestSentencesLemmatized = []
    bestSentencesOriginal = []
    if numberChosen == -1:
        numberChosen = len(sims)
    for index, value in sims[0:numberChosen]:
        bestIndexes.append(index)
        bestValues.append(value)
        sentenceLemmatized = General.readLineNumberFromFile(index, filenameLemmatized)
        sentenceOriginal = General.readLineNumberFromFile(index, filenameOriginal)
        bestSentencesLemmatized.append(sentenceLemmatized)
        bestSentencesOriginal.append(sentenceOriginal)
    return bestIndexes, bestValues, bestSentencesLemmatized, bestSentencesOriginal

def getStatisticsFromSimilarityMatrix(similarityMatrix):
    '''
    Input:
    Output:
    Description:
    Example:
    Related functions:
    '''
    arr = np.array([r for r in similarityMatrix])
    maxCurrent = -2
    minCurrent = 2
    nRows = nColumns = arr[0].size
    for i in range(nRows):
        for j in range(i + 1, nColumns):
            if (arr[i, j] > maxCurrent):
                maxCurrent = arr[i, j]
            if (arr[i, j] < minCurrent):
                minCurrent = arr[i, j]
    return arr, minCurrent, maxCurrent, np.mean(arr), np.std(arr), np.median(arr)

def getInverseDictFromDictionary(dictionary):
    '''
    @param dictionary: dictionary
    @type dictionary: C{dict}
    @return: Inverted dictionary
    @rtype: C{dict}
    '''
    # no idea why sometimes three inversions are needed, something related to the gensim datatype
    _map = {k: v for k, v in dictionary.items()}
    inv_map = {v: k for k, v in _map.items()}
    # _map = {v: k for k, v in inv_map.items()}
    return inv_map

def getDocumentsFromDictionaryAndCorpus(gensimDictionary, corpus):
    '''
    Input:
    Output:
    Description:
    Example:
    Related functions:
    '''
    if gensimDictionary.id2token:
        idTokenDict = gensimDictionary.id2token
    else:
        idTokenDict = getInverseDictFromDictionary(gensimDictionary.token2id)
    if isinstance(corpus, corpora.mmcorpus.MmCorpus):
        listCorpus = [corpus.docbyoffset(i) for i in corpus.index]
    if isinstance(corpus, list):
        listCorpus = corpus
    return [[x for elem in a for x in [idTokenDict[elem[0]]] * int(elem[1])] for a in listCorpus]

def createNewModelsFromOldModels(oldModelsName, newModelsName, numTopics, badWords, noBelow, noAbove):
    '''
    Creates and saves in disk a new model from an old one already present in disk. From the old model it extracts the docs and
    applies the models generating functions with new I{numTopics}, I{badWords}, I{noBelow} and I{noAbove} parameters.\n
    Example:\n
    C{num_topics = 100}\n
    C{badWords = ['dut', 'ni', 'zu', 'da', 'du', 'dute', 'zen', 'ere', 'gu', 'dugu', 'ez', 'bat', 'hori', 'hor', 'dira', 'baina', 'bi', 'zi', 'zut', 'zituzten', 'atzo', 'beste', 'dela']}\n
    C{noBelow = 5}\n
    C{noAbove = 5}\n
    C{createNewModelsFromOldModels('bertsoa', 'new', num_topics, badWords, noBelow, noAbove)}\n
    @param oldModelsName: oldModelsName
    @type oldModelsName: C{str}
    @param newModelsName: newModelsName
    @type newModelsName: C{str}
    @param numTopics: numTopics
    @type numTopics: C{int}
    @param badWords: badWords
    @type badWords: C{list}
    @param noBelow: noBelow
    @type noBelow: C{int}
    @param noAbove: noAbove
    @type noAbove: C{float}
    @return: None
    '''
    dictionaryOld, corpusOld, tfidfModelOld, lsiModelOld = loadPrecomputedData(oldModelsName)
    docs = getDocumentsFromDictionaryAndCorpus(dictionaryOld, corpusOld)
    dictionaryNew, corpusNew, tfidfModelNew, lsiModelNew = computePrunedData(docs, noBelow, noAbove, badWords, numTopics)
    savePrecomputedData(dictionaryNew, corpusNew, tfidfModelNew, lsiModelNew, newModelsName)
 
# def lemmatizeDocument(document):    
    '''
    document is already a clean document, without marks
    '''
#    return [Customize.lemmatize(w) for w in document.split()]

def lemmatizeListOfSentences(l, auxFilename = 'aux.txt'):
    '''
    '''
    General.saveListOfSentencesToFile(l, auxFilename)
    s = subprocess.check_output(['sh', 'ixa-pipe_eu.sh', auxFilename]) 
    s = s.split('\n')
    s = filter(None, s)
    return s

def lemmatizeString(s, auxFilename = 'aux.txt'):
    '''
    '''
    l = []
    l.append(s)
    return lemmatizeListOfSentences(l)[0]

def getListWordsFromListSentences(l):
    '''
    '''
    result = General.unlist([elem.split() for elem in l])
    return list(set(result))

def generateDocsFromFile(filename):
    '''
    '''
    l = General.readListOfSentencesFromFile(filename)
    return [elem.split() for elem in l]
