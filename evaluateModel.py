import numpy as np

def cosineSimilarity(embeddings, vector):
    rowNorms = np.linalg.norm(embeddings, axis=1)
    vectorNorm = np.linalg.norm(vector)
    rowNorms = np.where(rowNorms < 1e-8, 1.0, rowNorms)

    if vectorNorm < 1e-8:
        return np.zeros(len(embeddings))

    return embeddings @ vector / (rowNorms * vectorNorm)

def mostSimilar(
        centerEmbeddings,
        word,
        word2index,
        index2word,
        topN
):
    if word not in word2index:
        print(f"'{word}' not in vocabulary.")
        return []

    index = word2index[word]
    vector = centerEmbeddings[index]

    similarWordIndices = cosineSimilarity(centerEmbeddings, vector)

    similarWordIndices[index] = -2.0

    topIndices = np.argsort(similarWordIndices)[::-1][:topN]

    return [(index2word[i], float(similarWordIndices[i])) for i in topIndices]


def wordAnalogy(
        centerEmbeddings,
        a,
        b,
        c,
        word2index,
        index2word,
        topN
):
    for w in [a, b, c]:
        if w not in word2index:
            return f"'{w}' is not in vocabulary."

    query = centerEmbeddings[word2index[b]] - centerEmbeddings[word2index[a]] + centerEmbeddings[word2index[c]]

    similarWordIndices = cosineSimilarity(centerEmbeddings, query)

    for w in [a, b, c]:
        similarWordIndices[word2index[w]] = -2.0

    topIndices = np.argsort(similarWordIndices)[::-1][:topN]

    return [(index2word[i], float(similarWordIndices[i])) for i in topIndices]


def print_similar(results, query: str):
    print(f"\nMost similar to '{query}':")
    print(f"{'Word':<20} Similarity")
    print(f"{'-'*35}")
    for word, sim in results:
        print(f"{word:<20} {sim:.4f}")


def print_analogy(results, a, b, c):
    print(f"\n'{b}' - '{a}' + '{c}' = ?")
    print(f"{'Word':<20} Similarity")
    print(f"{'-'*35}")
    for word, sim in results:
        print(f"  {word:<20} {sim:.4f}")


if __name__ == "__main__":
    savePath = "skip-gram.npz"

    model = np.load(savePath)
    modelCenterEmbeddings = model["centerEmbeddings"]
    vocab = list(model["vocab"])
    word2index = {w: i for i, w in enumerate(vocab)}
    index2word = {i: w for i, w in enumerate(vocab)}

    for word in ["king", "paris", "computer", "music"]:
        results = mostSimilar(modelCenterEmbeddings, word, word2index, index2word, 8)
        if results:
            print_similar(results, word)

    for a, b, c in [("man", "king", "woman"), ("paris", "france", "berlin")]:
        results = wordAnalogy(modelCenterEmbeddings, a, b, c, word2index, index2word, 5)
        if isinstance(results, list):
            print_analogy(results, a, b, c)