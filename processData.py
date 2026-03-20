import numpy as np
from collections import Counter

def loadData(filepath, vocabSize=10000):
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

