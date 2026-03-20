from processData import *
from word2vec import Word2VecSkipGram


def trainModel(corpusFilepath, vocabSize, embeddingDim, epochs, subsampleThreshold, windowSize):

    wordIds, word2index, index2word, frequencies = loadData(corpusFilepath)

    model = Word2VecSkipGram(vocabSize, embeddingDim)

    for epoch in range(1, epochs + 1):
        corpus = subsample(wordIds, frequencies, subsampleThreshold)

        pairs = generateSkipgramPairs(corpus, windowSize)

