import word2vec
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from adjustText import adjust_text

def preprocess(model):
    idxs = []
    use_tag = ['JJ', 'NNP', 'NN', 'NNS']
    puncts = ["'",'>','<', '.', ':', ";", ',', "?", "!", u"’",'"']
    
    for i, label in enumerate(model.vocab):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tag and all(c not in label for c in puncts)):
            if label is not 'Page':
                idxs.append(i)
            
    vocabs = list(model.vocab[idxs])
    vectors = list(model.vectors[idxs])
    
    return vocabs, vectors

def plot_covers(used_words):
    color_array = np.arange(used_words)
    plt.figure(figsize=(15,8))
    plt.scatter(x_array, y_array, c=color_array,linewidths=0)
    texts = []
    for x, y, txt in zip(x_array, y_array, text_array):
        texts.append(plt.text(x, y, txt))
    return texts

data_path = 'data/all.txt'
model_path = 'data/test_model.bin'

word2vec.word2vec(data_path, model_path, size=80, window=8, alpha=0.05, iter_=3000)

model = word2vec.load(model_path)

vocabs, vectors = preprocess(model)

# train TSNE model
TRAIN_WORDS = 800
USED_WORDS = 80

tsne_model = TSNE(n_components=2, random_state=0, learning_rate=1000)
np.set_printoptions(suppress=True)
tsne_2_dimension = tsne_model.fit_transform(vectors[:TRAIN_WORDS])

x_array = tsne_2_dimension[:USED_WORDS,0]
y_array = tsne_2_dimension[:USED_WORDS,1]
text_array = vocabs[:USED_WORDS]

texts = plot_covers(USED_WORDS)
adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.show()