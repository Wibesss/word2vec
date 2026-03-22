# Word2Vec - Skip-gram Numpy implementation

This project is an implementation of **Word2Vec** **Skip-gram** model with **Negative Sampling** using only NumPy.

Model was trained on the text8 file as its dataset with vocabSize of 10000 and embeddingDim of 100, throughout 5 epochs
and a learningRate of 0.025. The complete training took around 10 hours after which the model shows a pretty good understanding of some words.
It is able to find good similarities between words and works well with word analogies.

In the continuation of this file you can read more about the way the model was implemented.

---
## Table of Contents
1. [Project Structure](#project-structure)
2. [Quickstart](#quickstart)
3. [Architecture](#architecture)
4. [Mathematical Derivation](#mathematical-derivation)
5. [Implementation Details](#implementation-details)
6. [The Dataset](#the-dataset)
7. [What I Would Improve With More Time](#what-i-would-improve-with-more-time)
8. [Final Words](#final-words)

---

## Project Structure

```
word2vec/
├── processData.py       — preprocessing, subsampling, pair generation, negative sampling
├── word2vec.py          — skip-gram forward pass, loss, gradients, parameter update, training step
├── tainModel.py         — training loop
├── evaluateModel.py     — cosine similarity, word analogies
└── text8                — text8 dataset
```

---

## Quickstart

```bash
# Install dependencies
pip install numpy

# Train skip-gram model
python trainModel.py

# Evaluate skip-gram model
python evaluateModel.py
```
---

## Architecture

### Skip-gram
If we have a **center** word skip-gram model is made to predict its surrounding **context** words.
The number of surrounding context words can be altered using the windowSize parameter.

For example:
```
"The cat sat on the mat"
      ↑
    center  →  predict:  ["The", "sat", "on"]
```
---

## Mathematical Derivation

### Notation

| Symbol     | Meaning                                             |
|------------|-----------------------------------------------------|
| $v_c$      | Center word vector (row of centerEmbeddings)        |
| $v_o$      | True context word vector (row of contextEmbeddings) |
| $v_k$      | Negative sample vector (row of $W_{out}$)           |
| $\sigma$   | Sigmoid function: $\sigma(x) = \frac{1}{1+e^{-x}}$  |
| $K$        | Number of negative samples                          |
| $\mathcal{L}$ | Loss function                                       |

---

### Loss Function (Negative Sampling)

Because a full softmax over a vocabulary of size $V$ is $O(V)$ makes the algoritham too slow we will use Negative sampling.  
Negative sampling replaces the full softmax with $K+1$ binary classifications where the loss function is calculated as:

$$\mathcal{L} = -\log \sigma(v_c \cdot v_o) \;-\; \sum_{k=1}^{K} \log \sigma(-v_c \cdot v_k)$$

- **Positive term**: the true context word should have a **high** dot product with the center (label = 1)  
- **Negative terms**: $K$ randomly sampled words should have a **low** dot product (label = 0)

Since $\log \sigma(-x) = \log(1 - \sigma(x))$ this is equivalent to binary cross-entropy and in code it looks like this:

```
positiveScore = self._sigmoid(vCenter @ vContext)
negativeScores = self._sigmoid(vKNegatives @ vCenter)

loss = (
    -np.log(positiveScore + 1e-10)
    -np.sum(np.log(1.0-negativeScores + 1e-10))
)
```

The epsilon terms 1e-10 are used to safeguard the sigmoids so that if the positiveScore or negativeScores
ever gets close to 0 or 1 respectivly, the sigmoid doesn't blow up and go to infinity, so with the terms even if by some
chance that is true the value being added will offset that mistake.

---

### Gradient Derivations

Two facts that are repeatedly used to get to the needed gradient formulas are:

$$\frac{d}{dx}\log\sigma(x) = 1 - \sigma(x) \qquad \frac{d}{dx}\log\sigma(-x) = -\sigma(x)$$

Combined with the chain rule and the dot-product gradient rule $\frac{\partial}{\partial a}(a \cdot b) = b$.

To get the updated values of each vector we partially derive the loss 
function with respect to that specific vector and get the gradient for it

---

**When we take the partial derivative of the loss with respect to $v_c$ we get:**

#### $$\frac{\partial \mathcal{L}}{\partial v_c} = \frac{\partial}{\partial v_c}(-\log \sigma(v_c \cdot v_o) \;-\; \sum_{k=1}^{K} \log \sigma(-v_c \cdot v_k))$$


From deriving the positive term we get:

$$\frac{\partial}{\partial v_c}\bigl[-\log\sigma(v_c \cdot v_o)\bigr]
= -(1 - \sigma(v_c \cdot v_o)) \cdot v_o
= (\sigma(v_c \cdot v_o) - 1) \cdot v_o$$

While from the $k$-th negative term we get:

$$\frac{\partial}{\partial v_c}\bigl[-\log\sigma(-v_c \cdot v_k)\bigr]
= \sigma(v_c \cdot v_k) \cdot v_k$$

In the end by combining these do we get the full gradient:

$$\boxed{\frac{\partial \mathcal{L}}{\partial v_c}
= \bigl(\sigma(v_c \cdot v_o) - 1\bigr)\,v_o
\;+\; \sum_{k=1}^{K} \sigma(v_c \cdot v_k)\,v_k}$$

Code implementation:
```
positiveScore = self._sigmoid(vCenter @ vContext)
negativeScores = self._sigmoid(vKNegatives @ vCenter)
gradientVCenter = (positiveScore-1.0) * vContext + np.sum(negativeScores[:,None] * vKNegatives, axis=0)
```

---
**When we take the partial derivative of the loss with respect to $v_o$ we get:**

#### $$\frac{\partial \mathcal{L}}{\partial v_o} = \frac{\partial}{\partial v_o}(-\log \sigma(v_c \cdot v_o) \;-\; \sum_{k=1}^{K} \log \sigma(-v_c \cdot v_k))$$

Only the positive term involves $v_o$ so the full gradient is:

$$\boxed{\frac{\partial \mathcal{L}}{\partial v_o}
= \bigl(\sigma(v_c \cdot v_o) - 1\bigr)\,v_c}$$

Code implementation:
```
positiveScore = self._sigmoid(vCenter @ vContext)
gradientVContext = (positiveScore - 1.0) * vCenter
```

---
**When we take the partial derivative of the loss with respect to $v_k$ we get:**

#### $$\frac{\partial \mathcal{L}}{\partial v_k} = \frac{\partial}{\partial v_k}(-\log \sigma(v_c \cdot v_o) \;-\; \sum_{k=1}^{K} \log \sigma(-v_c \cdot v_k))$$

Each negative sample's term involves only its own $v_k$ which means the full gradient is:

$$\boxed{\frac{\partial \mathcal{L}}{\partial v_k}
= \sigma(v_c \cdot v_k)\,v_c}$$

Code implementation:
```
negativeScores = self._sigmoid(vKNegatives @ vCenter)
gradientVKNegatives = negativeScores[:,None] * vCenter
```

---

### Intuition: Why does $(σ - 1)$ make sense?

The term $(\sigma(v_c \cdot v_o) - 1) \in (-1, 0]$.

- When the model is **wrong** ($\sigma \approx 0$): gradient $\approx -1 \cdot v_o$ → **large update**, push vectors together  
- When the model is **right** ($\sigma \approx 1$): gradient $\approx 0$ → **small update**, already good

This is exactly the behavior we want. We want the model to learn more from mistakes and guide it back to the solution that is more correct.

---

## Implementation Details

Some of the useful optimization that are used in this project are:

### Subsampling of Frequent Words
Words like "the" and "a" carry little semantic signal so they are discarded during training with probability:

$$P(\text{discard } w) = 1 - \sqrt{\frac{t}{f(w)}}, \quad t = 10^{-5}$$

This speeds up training and improves embeddings for content words.

Code implementation of the subsampling looks like this:

```
def subsample(wordIds, frequencies, subsampleThreshold):
    f = frequencies[wordIds]
    keepProbabilities = np.sqrt(subsampleThreshold / (f + 1e-10))
    keepProbabilities = np.clip(keepProbabilities, 0.0, 1.0)

    mask = np.random.random(len(wordIds)) < keepProbabilities

    return wordIds[mask].tolist()
```

### Negative Sampling Distribution
Negative words are sampled proportional to $f(w)^{0.75}$.  
The exponent 0.75 (from the original paper) dampens the dominance of very frequent words without making the distribution uniform.

Code implementation:
```
def getNegativeProbabilities(frequencies):
    negativeProbabilities = frequencies ** 0.75
    negativeProbabilities /= np.sum(negativeProbabilities)
    return negativeProbabilities
```

this is then later used every time we are sampling the negatives.

### Learning Rate Schedule
Linear decay from `learningRate` to `learningRateMin`:

$$\text{lr} = \text{learningRate} \times \left(1 - \frac{\text{step}}{\text{totalSteps}}\right)$$



in code this is calculated like this:

```
progress = ((epoch-1) * totalPairs + step) / (epochs * totalPairs)
learningRate = max(learningRateStart * (1.0 - progress), learningRateMin)
```

---

## The Dataset

The model was trained on the text8 file, which is a widely used in natural language processing and machine learning.
It is a cleaned dataset for training algorithms on text generation, word emeddings and compression.

It cointains the first 100 milion characters of the March 2006 Wikipedia XML dump which has been heavily 
preprocessed into a single line, where all the non-Latin characters, punctuation, and markups have been removed

The size of the file is roughly 100mb and it is still being maintained to this day.

---

## What I Would Improve With More Time

- **Hierarchical Softmax** — alternative to negative sampling using a Huffman tree; theoretically cleaner, often faster for large vocabularies because the more frequent words would be encoded with shorter codes 
- **Batched training** — vectorise over multiple pairs simultaneously for significant NumPy speedup  
- **Evaluation on standard benchmarks** — I would like to see how the model stacks up against other similar models and find more ways to improve it WordSim-353, SimLex-999, Google Analogy Test Set  

---
## Final Words

Word2Vec demonstrates a core principle: **useful representations emerge 
from structure in raw text**, with no labels required. By training a model 
to predict context from a center word, the embeddings organically encode 
semantic relationships — synonyms cluster together, analogies become vector 
arithmetic, and meaning becomes geometry.

The same self-supervised intuition scales directly to modern LLMs: BERT's 
masked language model and GPT's next-token prediction are both descendants 
of this same idea. Word2Vec is where it started.