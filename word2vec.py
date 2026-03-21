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
        loss, backwardCache, updateCache = self.forward(center, context, negatives)
        gradients = self.backward(backwardCache)
        gradientVCenter, gradientVContext, gradientVKNegatives = gradients
        self.update(updateCache, gradientVCenter, gradientVContext, gradientVKNegatives, learningRate)
        return loss


    def forward(self, center, context, negatives):

        vCenter = self.centerEmbeddings[center]
        vContext = self.contextEmbeddings[context]
        vKNegatives = self.contextEmbeddings[negatives]

        positiveScore = self._sigmoid(vCenter @ vContext)
        negativeScores = self._sigmoid(vKNegatives @ vCenter)

        loss = (
            -np.log(positiveScore + 1e-10)
            -np.sum(np.log(1.0-negativeScores+1e-10))
        )

        backwardCache = (vCenter, vContext, vKNegatives, positiveScore, negativeScores)
        updateCache = (center, context, negatives)

        return float(loss), backwardCache, updateCache


    def backward(self, backwardCache):
        vCenter, vContext, vKNegatives, positiveScore, negativeScores = backwardCache

        gradientVCenter = (positiveScore-1.0) * vContext + np.sum(negativeScores[:,None] * vKNegatives, axis=0)

        gradientVContext = (positiveScore - 1.0) * vCenter

        gradientVKNegatives = negativeScores[:,None] * vCenter

        return gradientVCenter, gradientVContext, gradientVKNegatives


    def update(self, updateCache, gradientVCenter, gradientVContext, gradientVKNegatives, learningRate):
        center, context, negatives = updateCache

        self.centerEmbeddings[center] -= learningRate * gradientVCenter
        self.contextEmbeddings[context] -= learningRate * gradientVContext
        self.contextEmbeddings[negatives] -= learningRate * gradientVKNegatives
