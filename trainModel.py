import time
import numpy as np

from processData import loadData, subsample, generateSkipgramPairs, getNegativeProbabilities, sampleNegatives
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
LOG_EVERY = 100000
SAVE_PATH = "skip-gram.npz"

def trainModel(
        corpusFilepath,
        vocabSize,
        embeddingDim,
        epochs,
        subsampleThreshold,
        windowSize,
        learningRateStart,
        learningRateMin,
        kNegatives,
        logEvery,
        savePath,
):
    print(f"\n{'=' * 55}")
    print(f"Skip-gram Word2Vec")
    print(f"{'=' * 55}\n")

    wordIds, word2index, index2word, frequencies = loadData(corpusFilepath, vocabSize)

    model = Word2VecSkipGram(vocabSize, embeddingDim)

    print(f"[model] vocab={len(word2index)}  embed_dim={embeddingDim} params={2 * len(word2index) * embeddingDim}\n")

    negativeProbabilities = getNegativeProbabilities(frequencies)

    epochLosses = []


    for epoch in range(1, epochs + 1):
        corpus = subsample(wordIds, frequencies, subsampleThreshold)

        pairs = list(generateSkipgramPairs(corpus, windowSize))
        totalPairs = len(pairs)

        running_loss = 0.0
        epoch_start = time.time()

        for step, (center, context) in enumerate(pairs, start=1):
            progress = ((epoch-1) * totalPairs + step) / (epochs * totalPairs)
            learningRate = max(learningRateStart * (1.0 - progress), learningRateMin)

            exclude = {center, context}
            negatives = sampleNegatives(negativeProbabilities, exclude, kNegatives)

            loss = model.trainStep(center, context, negatives, learningRate)

            running_loss += loss

            if step % logEvery == 0:
                avg = running_loss / step
                elapsed = time.time() - epoch_start
                percent = 100 * step / totalPairs
                print(f"epoch {epoch}/{epochs} [{percent:5.1f}%] loss={avg:.4f} lr={learningRate:.6f} elapsed={elapsed:.0f}s")

        avgLoss = running_loss / totalPairs
        epochLosses.append(avgLoss)
        elapsed = time.time() - epoch_start
        print(f"\nEpoch {epoch} complete │ avg loss={avgLoss:.4f} │ time={elapsed:.0f}s ──\n")

    np.savez(
        savePath,
        centerEmbeddings = model.centerEmbeddings,
        contextEmbeddings = model.contextEmbeddings,
        vocab=list(index2word.values()),
    )
    print(f"[save] Embeddings saved to '{savePath}'")

    return model, word2index, index2word


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
        K_NEGATIVES,
        LOG_EVERY,
        SAVE_PATH,
    )
