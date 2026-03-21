from processData import *
from word2vec import Word2VecSkipGram

CORPUS_FILEPATH = "text8"
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
EPOCHS = 5
SUBSAMPLE_THRESHOLD = 1e-5
WINDOW_SIZE = 5
LEARNING_RATE_START = 0.025
LEARNING_RATE_MIN = 0.001
K_NEGATIVES = 5

def trainModel(
        corpusFilepath,
        vocabSize,
        embeddingDim,
        epochs,
        subsampleThreshold,
        windowSize,
        learningRateStart,
        learningRateMin,
        kNegatives
):

    wordIds, word2index, index2word, frequencies = loadData(corpusFilepath, vocabSize)

    model = Word2VecSkipGram(vocabSize, embeddingDim)

    for epoch in range(1, epochs + 1):
        corpus = subsample(wordIds, frequencies, subsampleThreshold)

        pairs = list(generateSkipgramPairs(corpus, windowSize))
        totalPairs = len(pairs)

        negativeProbabilities = getNegativeProbabilities(frequencies)

        for step, (center, context) in enumerate(pairs, start=1):
            progress = ((epoch-1) * totalPairs + step) / (epochs * totalPairs)
            learningRate = max(learningRateStart * (1.0 - progress), learningRateMin)

            exclude = {center, context}
            negatives = sampleNegatives(negativeProbabilities, exclude, kNegatives)

            loss = model.trainStep(center, context, negatives, learningRate)





if __name__ == "__main__":

    trainModel(
        CORPUS_FILEPATH,
        VOCAB_SIZE,
        EMBEDDING_DIM,
        EPOCHS,
        SUBSAMPLE_THRESHOLD,
        WINDOW_SIZE,
        LEARNING_RATE_START,
        LEARNING_RATE_MIN,
        K_NEGATIVES
    )
