import numpy as np
from collections import Counter

def loadData(filepath, vocabSize):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    words = text.split()
    frequenciesCounter = Counter(words)

    mostCommon = [word for word, _ in frequenciesCounter.most_common(vocabSize -1)]

    vocab = ["<UNK>"] + mostCommon

    word2index = {word: i for i, word in enumerate(vocab)}
    index2word = {i: word for i, word in enumerate(vocab)}

    wordIds = np.array([word2index.get(word, 0) for word in words])

    wordCounts = np.array([frequenciesCounter.get(word, 0) for word in vocab])

    frequencies = wordCounts / np.sum(wordCounts)

    return wordIds, word2index, index2word, frequencies


def subsample(wordIds, frequencies, subsampleThreshold):
    f = frequencies[wordIds]
    keepProbabilities = np.sqrt(subsampleThreshold / (f + 1e-10))
    keepProbabilities = np.clip(keepProbabilities, 0.0, 1.0)

    mask = np.random.random(len(wordIds)) < keepProbabilities

    return wordIds[mask].tolist()


def generateSkipgramPairs(corpus, windowSize):
    for i, center in enumerate(corpus):
        window = np.random.randint(1, windowSize+1)
        start = max(0, i - window)
        end = min(i+window+1, len(corpus))
        for j in range(start, end):
            if j!=i:
                yield center,corpus[j]



