

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np



# ======================
# 1. Load GloVe
# ======================


def load_glove(path):
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=float)
            embeddings[word] = vector
    return embeddings


glove = load_glove("data/glove_50k_50d.txt")

# ======================
# 2. Cosine Similarity
# ======================
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_similar_words(word, glove, top_n=40):
    if word not in glove:
        return []

    target_vec = glove[word]
    similarities = []

    for w, vec in glove.items():
        sim = cosine_similarity(target_vec, vec)
        similarities.append((w, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in similarities[1:top_n+1]]

# ======================
# 3. Build categories
# ================== ===
categories = {
    "sports": get_similar_words("football", glove, 40),
    "technology": get_similar_words("computer", glove, 40),
    "finance": get_similar_words("money", glove, 40),
    "emotions": get_similar_words("happy", glove, 40),
    "countries": get_similar_words("france", glove, 40),
}

# ======================
# 4. Convert to vectors
# =====================
words = []
labels = []
vectors = []

for category, word_list in categories.items():
    for word in word_list:
        if word in glove:
            words.append(word)
            labels.append(category)
            vectors.append(glove[word])

X = np.array(vectors)

print("Total words:", len(words))

# ======================
# 5. t-SNE
# ======================
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)

# ======================
# 6. Plot
# ======================
plt.figure(figsize=(12, 10))

for category in set(labels):
    idx = [i for i, l in enumerate(labels) if l == category]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=category)

# ======================
# 7. Annotation
# ======================
important_words = ["football", "computer", "money", "happy", "france"]

for word in important_words:
    if word in words:
        i = words.index(word)
        plt.annotate(word, (X_2d[i, 0], X_2d[i, 1]))

plt.legend()
plt.title("Word Embeddings (t-SNE)")
plt.savefig("outputs/word_plot.png")
plt.show()
# print(glove['computer'][:5])


