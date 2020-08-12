# review_data.py

from utils import *
import re
import json
import numpy as np


# Wraps a sequence of word indices with a 1-5 label (5 star ratings)
class ReviewExample:
    def __init__(self, indexed_words, star):
        self.indexed_words = indexed_words
        self.star = star

    def __repr__(self):
        return repr(self.indexed_words) + "; star=" + repr(self.star)

    def get_indexed_words_reversed(self):
        return [self.indexed_words[len(self.indexed_words) - 1 - i] for i in xrange(0, len (self.indexed_words))]


# Reads review examples in the json format; 
# tokenizes and indexes the (1st and last) sentence according
# to the vocabulary in indexer. If add_to_indexer is False, replaces unseen words with UNK, otherwise grows the
# indexer. word_counter optionally keeps a tally of how many times each word is seen (mostly for logging purposes).
def read_and_index_review_examples(infile, indexer, review_count, add_to_indexer=False, word_counter=None):
    f = open(infile)
    exs = []
    count = 0
    for line in f:
        review = json.loads(line)
        star = review['stars']
        #temp = review['text'].splt("\n")
        sents = review['text'].lower().split('.')
        #sents = review['text'].split(".") # sentences
        # take the first and last sentence (sents[-1] is '', use sents[-2])
        if len(sents) > 0:
            if len(sents) > 2:
                tokenized_cleaned_sent = filter(lambda x: x != '', clean_str(sents[0]+ ' .').rstrip().split(" ")) # + sents[-2]
            else:
                tokenized_cleaned_sent = filter(lambda x: x != '', clean_str(sents[0]+ ' .').rstrip().split(" "))
                
            if word_counter is not None:
                for word in tokenized_cleaned_sent:
                    word_counter.increment_count(word, 1.0)
            indexed_sent = [indexer.get_index(word) if indexer.contains(word) or add_to_indexer else indexer.get_index("UNK")
                 for word in tokenized_cleaned_sent]
            exs.append(ReviewExample(indexed_sent, star))
            if count >= review_count:
                break
            else:
                count +=1
    f.close()
    return exs

# Writes review examples to an output file in the same format they are read in. However, note that what gets written
# out is tokenized and contains UNKs, so this will not exactly match the input file.
def write_review_examples(exs, outfile, indexer):
    o = open(outfile, 'w')
    for ex in exs:
        o.write(repr(ex.star) + "\t" + " ".join([indexer.get_object(idx) for idx in ex.indexed_words]) + "\n")
    o.close()


# Tokenizes and cleans a string: contractions are broken off from their base words, punctuation is broken out
# into its own token, junk characters are removed, etc.
def clean_str(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`\-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\-", " - ", string)
    # We may have introduced double spaces, so collapse these down
    string = re.sub(r"\s{2,}", " ", string)
    return string


# Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
# word in the indexer. The 0 vector is returned if an unknown word is queried.
class WordEmbeddings:
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding(self, word):
        word_idx = self.word_indexer.index_of(word)
        #word_idx = self.word_indexer.get_index(word)#??add = False? or index_of, returns -1 if word not present
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[word_indexer.get_index("UNK")]


# Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
# that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
# word embedding files.
def read_word_embeddings(embeddings_file):
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx+1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            #print repr(float_numbers)
            vector = np.array(float_numbers)
            word_indexer.get_index(word)
            vectors.append(vector)
            #print repr(word) + " : " + repr(vector)
    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    # Add an UNK token at the end
    word_indexer.get_index("UNK")
    vectors.append(np.zeros(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return WordEmbeddings(word_indexer, np.array(vectors))


#################
# You probably don't need to interact with this code unles you want to relativize other sets of embeddings
# to this data. Relativization = restrict the embeddings to only have words we actually need in order to save memory
# (but this requires looking at the data in advance).

# Relativize the word vectors to the training set
def relativize(file, outfile, indexer, word_counter):
    f = open(file)
    o = open(outfile, 'w')
    voc = []
    for line in f:
        word = line[:line.find(' ')]
        if indexer.contains(word):
            print("Keeping word vector for " + word)
            voc.append(word)
            o.write(line)
    for word in indexer.objs_to_ints.keys():
        if word not in voc:
            print("Missing " + word + " with count " + repr(word_counter.get_count(word)))
    f.close()
    o.close()
    
# ----Example----
# word_vectors = read_word_embeddings("data/glove.6B.300d-relativized.txt")
# train_exs = read_and_index_review_examples("data/review1.json", word_vectors.word_indexer)
