import numpy as np

class Word2VecSkipGram:
    def __init__(self, vocabSize, embeddingDim):
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim

    def trainStep(self, center, context, negatives, learningRate):
