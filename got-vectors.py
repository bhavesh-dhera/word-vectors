from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import os
import glob
import sklearn.manifold
import multiprocessing
import gensim.models.word2vec as w2v
import codecs
import nltk
import re
import matplotlib.pyplot as plt
#ltk.download("punkt")
#nltk.download("stopwords")

#make books one string
book_filenames = sorted(glob.glob("./hp/*.txt"))
print(book_filenames)

corpus_raw = u""
for book_filename in book_filenames:
	print("Reading '{0}' ...".format(book_filename))
	with codecs.open(book_filename, "r", "utf-8", errors="ignore") as book_file:
		corpus_raw += book_file.read()
	print("Corpus is now {0} characters long".format(len(corpus_raw)))

wordd= len(corpus_raw)
#split corpus into sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

#convert into list of words
def sentence_to_worldlist(raw):
	clean = re.sub("[^a-zA-Z]"," ", raw)
	words = clean.split()
	return words

sentences = []
for raw_sentence in raw_sentences:
	if len(raw_sentence) > 0:
		sentences.append(sentence_to_worldlist(raw_sentence))

#example
print(raw_sentences[5])
print(sentence_to_worldlist(raw_sentences[5]))

#TRAINING W2V MOdel

num_features = 300 #no of dimensions
min_word_count = 3 #min word count
num_workers = multiprocessing.cpu_count() #cores
context_size = 7 #windows size

downsampling = 1e-3

seed =1
t2v = w2v.Word2Vec(sg=1,seed=seed,workers=num_workers,size=num_features
 	,min_count = min_word_count, window = context_size, sample = downsampling)

t2v.build_vocab(sentences)
print('Training....')
t2v.train(sentences, total_examples = t2v.corpus_count, epochs = t2v.iter)


#to visualise 300D graph as 2-D graph

print('Entering TSNE..')
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = t2v.wv.syn0
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

print('Plotting....')
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[t2v.wv.vocab[word].index])
            for word in t2v.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

points.plot.scatter("x", "y", s=10, figsize=(20, 12))
plt.show()


print(t2v.most_similar("Dumbledore"))
print(t2v.most_similar("Patronus"))
print()
print()
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = t2v.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

def nearest_similarity_cosmul_1(start1, end1, start2):
    similarities = t2v.most_similar_cosmul(
        positive=[start1],
        negative=[end1,start2]
    )
    end2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

nearest_similarity_cosmul("Harry","Voldemort","Filch")
#nearest_similarity_cosmul("Harry","Shae","Cersei")
nearest_similarity_cosmul_1("Harry","Cho","Ron")
