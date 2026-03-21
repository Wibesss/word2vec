import numpy as np

class Word2VecSkipGram:
    def __init__(self, vocabSize, embeddingDim):
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim

        self.centerEmbeddings = np.random.randn(vocabSize, embeddingDim).astype(np.float32) * 0.01
        self.contextEmbeddings = np.random.randn(vocabSize, embeddingDim).astype(np.float32)* 0.01

    @staticmethod
    def _sigmoid(x):
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )


    def trainStep(self, center, context, negatives, learningRate):
        loss = self.forward(center, context, negatives)


    def forward(self, center, context, negatives):

        vCenter = self.centerEmbeddings[center]
        vContext = self.contextEmbeddings[context]
        vKNegatives = self.contextEmbeddings[negatives]

        positiveScore = self._sigmoid(vCenter @ vContext)
        negativeScores = self._sigmoid(vKNegatives @ vContext)

        loss = (
            -np.log(positiveScore + 1e-10)
            -np.sum(np.log(1.0-negativeScores+1e-10))
        )

        return float(loss)